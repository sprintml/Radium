"""Submit detection-only re-runs (with save_features=true) for every (m1, mi,
frac) combo declared for one watermark in configs/experiments.yaml.

For each watermark the planner emits two row types:

  * One ``<wm>_m1-<m1>_clean_savefeatures`` row per (watermark, m1). Its
    ``run.chain.stages`` is restricted to detection_m1_supervised /
    _unsupervised / _watermarked. M1 detection paths
    (``M1_<m1>/clean/<dataset>/<set>/evaluations/...``) are keyed by
    (m1, dataset, set) only, so collapsing to one row guarantees the M1
    clean detection slurm job is submitted once per (watermark, m1).

  * One ``<wm>_m1-<m1>_mi-<mi>_f<frac>_<src>_savefeatures`` row per
    (mi, frac). Its ``run.chain.stages`` is restricted to
    detection_mi_supervised / _unsupervised — Mi-only, no M1 work.

Both row types set run.detection_mode=aggregation and
detection.save_features=true on the stages they include, so each detection
stage writes ``<output_dir>/features/<wm>.npz`` alongside the scores cache.

The Mi rows do NOT depend on the M1 row at submit time (Mi detection reads
its own generation_mi outputs from disk and is independent of the M1 score
cache for this stage). They can run in parallel.
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

from plan_experiments import (  # noqa: E402
    WM_DIR,
    _resolve_fracs,
    _row_for_main,
)
from pipeline.hyperparameter_sweep import (  # noqa: E402
    Row,
    runs_with_active_jobs,
    submit_rows,
)


M1_DETECTION_STAGES = [
    "detection_m1_supervised",
    "detection_m1_unsupervised",
    "detection_m1_watermarked",
]
MI_DETECTION_STAGES = [
    "detection_mi_supervised",
    "detection_mi_unsupervised",
]
COMMON_DETECT_OVERRIDES = [
    ("run.chain.start_i", "1"),
    ("run.chain.end_i", "1"),
    ("run.detection_mode", "aggregation"),
]


def _wm_name(entry: dict) -> str:
    wm_yaml = entry.get("wm_yaml")
    if wm_yaml:
        return Path(wm_yaml).stem
    return entry["wm"]


def _expand_for_watermark(entries: list[dict], aliases: dict, watermark: str) -> list[Row]:
    """Build two row types per (watermark, m1):
      * one ``..._clean_savefeatures`` row — only detection_m1_* stages.
      * one ``..._mi-<mi>_f<frac>_..._savefeatures`` row per (mi, frac) —
        only detection_mi_* stages.

    M1 detection writes to a path keyed by (m1_model_type, dataset, set_name)
    only, so collapsing it to one row prevents every Mi row from queueing
    its own redundant M1 job.
    """
    matching = [e for e in (entries or []) if _wm_name(e) == watermark]
    if not matching:
        return []

    rows: list[Row] = []
    seen_clean: set[str] = set()
    for entry in matching:
        mis = entry.get("mi")
        if not mis:
            raise ValueError(f"main entry missing 'mi': {entry!r}")
        if isinstance(mis, str):
            mis = [mis]
        fracs = _resolve_fracs(entry["fracs"], aliases)
        m1 = entry["m1"]

        if m1 not in seen_clean:
            seen_clean.add(m1)
            rows.append(_clean_row(entry, mis[0]))

        for mi in mis:
            for frac in fracs:
                base = _row_for_main(entry, mi, frac)
                rows.append(_mi_row(base))
    return rows


def _save_features_overrides(stages: list[str]) -> list[tuple[str, str]]:
    return [
        (f"stages.{s}.chain_overrides.detection.save_features", "true")
        for s in stages
    ]


def _clean_row(entry: dict, mi: str) -> Row:
    """One detection_m1_* row per (watermark, m1).

    `mi` only matters for the planner to resolve a finetuning_config; it's not
    used by the M1 detection stages themselves. We pick the first declared mi
    for the entry so we get a valid finetuning yaml.
    """
    base = _row_for_main(entry, mi, 0.0)
    wm_name = _wm_name(entry)
    name = f"{wm_name}_m1-{entry['m1']}_clean_savefeatures"
    overrides = list(base.overrides)
    overrides.extend(COMMON_DETECT_OVERRIDES)
    overrides.append(("run.chain.stages", "[" + ",".join(M1_DETECTION_STAGES) + "]"))
    overrides.extend(_save_features_overrides(M1_DETECTION_STAGES))
    return Row(name, overrides)


def _mi_row(base: Row) -> Row:
    name = f"{base.name}_savefeatures"
    overrides = list(base.overrides)
    overrides.extend(COMMON_DETECT_OVERRIDES)
    overrides.append(("run.chain.stages", "[" + ",".join(MI_DETECTION_STAGES) + "]"))
    overrides.extend(_save_features_overrides(MI_DETECTION_STAGES))
    return Row(name, overrides)


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
    p.add_argument("--watermark", required=True,
                   help="Watermark name (matches `wm:` field or wm_yaml stem in experiments.yaml)")
    p.add_argument("--config", default="configs/experiments.yaml",
                   help="Path to experiments yaml (relative to repo root or absolute)")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open() as f:
        spec = yaml.safe_load(f) or {}

    aliases = spec.get("fraction_aliases") or {}
    rows = _dedupe(_expand_for_watermark(spec.get("main", []), aliases, args.watermark))

    if not rows:
        print(f"No main entries match watermark={args.watermark!r}.")
        return

    user = os.environ.get("USER", "")
    active = runs_with_active_jobs(user)
    skip_slurm = [r.name for r in rows if r.name in active]
    pending = [r for r in rows if r.name not in active]

    print("============== Plan detect (save_features) ==============")
    print(f"Spec:                {cfg_path}")
    print(f"Watermark:           {args.watermark}")
    print(f"Cluster:             {args.cluster}")
    print(f"Total rows:          {len(rows)}")
    print(f"Skip (active slurm): {len(skip_slurm)}")
    print(f"Pending:             {len(pending)}")
    print("=========================================================\n", flush=True)

    if not pending:
        print("Nothing to submit.")
        return

    submit_rows(pending, cluster=args.cluster)


if __name__ == "__main__":
    main()
