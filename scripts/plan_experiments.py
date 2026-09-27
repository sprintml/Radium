"""Submit experiments declared in configs/experiments.yaml.

Reads the spec, expands `main` rows to (wm, m1, mi, frac, src) tuples
and `ablations` entries to OFAT rows from scripts/run_ablations.py, then
delegates submission to pipeline.hyperparameter_sweep.submit_rows (which
in turn shells out to pipeline/run_pipeline.py per row).

Skip rule (shallowest possible): a row is skipped if either
  - {RADIOACTIVITY_OUT}/pipeline_configs/{run_name}/ exists on disk, OR
  - a slurm job for the current user has its StdOut under
    .../slurm_out/{run_name}/

To re-submit a failed run, rm its pipeline_configs/{run_name}/ directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline.hyperparameter_sweep import (  # noqa: E402
    Row,
    plan_rows,
    runs_with_active_jobs,
    submit_rows,
)


WM_DIR = "configs/watermarks"
DEFAULT_SRC = "generated"


def _ft_for_mi(mi: str) -> str:
    if mi.startswith("infinity"):
        return "configs/finetuning/infinity/infinity_2b.yaml"
    if mi.startswith("sd"):
        return "configs/finetuning/stablediffusion/common.yaml"
    raise ValueError(f"Unknown mi model type: {mi!r}")


def _gpu_profile_for_mi(mi: str) -> str:
    return "high_vram" if mi.startswith("infinity") else "default"


def _frac_str(f: float) -> str:
    """Encode a fraction the same way run_pipeline.sh / scripts/_planned_to_submit.py
    did: '0', '1.0', else %g."""
    if f == 0:
        return "0"
    if f == 1.0:
        return "1.0"
    return f"{f:g}"


def _resolve_fracs(spec, aliases: dict) -> list[float]:
    if isinstance(spec, str):
        if spec not in aliases:
            raise KeyError(f"Unknown fraction alias: {spec!r}")
        return [float(x) for x in aliases[spec]]
    return [float(x) for x in spec]


def _row_for_main(entry: dict, mi: str, frac: float) -> Row:
    wm_yaml = entry.get("wm_yaml")
    if wm_yaml:
        wm_name = Path(wm_yaml).stem
    else:
        wm_name = entry["wm"]
        wm_yaml = f"{WM_DIR}/{wm_name}.yaml"
    src = entry.get("src", DEFAULT_SRC)
    suffix = entry.get("name_suffix")
    m1 = entry["m1"]
    frac_s = _frac_str(frac)

    name = f"{wm_name}_m1-{m1}_mi-{mi}_f{frac_s}_{src}"
    if suffix:
        name = f"{name}_{suffix}"

    overrides: list[tuple[str, str]] = [
        ("run.watermark_config", wm_yaml),
        ("run.m1_model_type", m1),
        ("run.mi_model_type", mi),
        ("run.finetuning_config", _ft_for_mi(mi)),
        ("stages.finetuning.gpu_profile", _gpu_profile_for_mi(mi)),
        ("stages.finetuning.chain_overrides.training.watermark_fraction", frac_s),
        ("stages.finetuning.chain_overrides.datasets.finetune_clean_source", src),
    ]
    for k, v in (entry.get("overrides") or {}).items():
        overrides.append((k, str(v)))
    return Row(name, overrides)


def _expand_main(entries: list[dict], aliases: dict) -> list[Row]:
    rows: list[Row] = []
    for entry in entries or []:
        mis = entry.get("mi")
        if not mis:
            raise ValueError(f"main entry missing 'mi': {entry!r}")
        if isinstance(mis, str):
            mis = [mis]
        fracs = _resolve_fracs(entry["fracs"], aliases)
        for mi in mis:
            for frac in fracs:
                rows.append(_row_for_main(entry, mi, frac))
    return rows


def _expand_ablations(entries: list[dict]) -> list[Row]:
    if not entries:
        return []
    from run_ablations import CAMPAIGNS  # imported lazily
    rows: list[Row] = []
    for entry in entries:
        name = entry["campaign"]
        if name not in CAMPAIGNS:
            raise KeyError(f"Unknown ablation campaign: {name!r}. "
                           f"Known: {sorted(CAMPAIGNS)}")
        cells, axes_for_fn = CAMPAIGNS[name]
        for cell in cells:
            rows.extend(plan_rows(cell, axes_for_fn(cell)))
    return rows


def _existing_on_disk(out_base: Path, name: str) -> bool:
    return (out_base / "pipeline_configs" / name).is_dir()


def _dedupe(rows: list[Row]) -> list[Row]:
    seen: set[str] = set()
    out: list[Row] = []
    for r in rows:
        if r.name in seen:
            continue
        seen.add(r.name)
        out.append(r)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/experiments.yaml",
                   help="Path to experiments yaml (relative to repo root or absolute)")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open() as f:
        spec = yaml.safe_load(f) or {}

    aliases = spec.get("fraction_aliases") or {}
    main_rows = _expand_main(spec.get("main", []), aliases)
    abl_rows = _expand_ablations(spec.get("ablations", []))
    rows = _dedupe(main_rows + abl_rows)

    out_base = Path(os.environ.get(
        "RADIOACTIVITY_OUT",
        str(PROJECT_ROOT.parent / "output"),
    )).resolve()
    user = os.environ.get("USER", "")
    active = runs_with_active_jobs(user)

    skip_disk: list[str] = []
    skip_slurm: list[str] = []
    pending: list[Row] = []
    for row in rows:
        if row.name in active:
            skip_slurm.append(row.name)
            continue
        if _existing_on_disk(out_base, row.name):
            skip_disk.append(row.name)
            continue
        pending.append(row)

    print("==================== Plan experiments ====================")
    print(f"Spec:                {cfg_path}")
    print(f"Output root:         {out_base}")
    print(f"Cluster:             {args.cluster}")
    print(f"Total rows:          {len(rows)} ({len(main_rows)} main, {len(abl_rows)} ablation)")
    print(f"Skip (on disk):      {len(skip_disk)}")
    print(f"Skip (active slurm): {len(skip_slurm)}")
    print(f"Pending:             {len(pending)}")
    print("==========================================================\n", flush=True)

    if not pending:
        print("Nothing to submit.")
        return

    submit_rows(pending, cluster=args.cluster)


if __name__ == "__main__":
    main()
