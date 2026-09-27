"""Ablation sweep orchestration on top of run_pipeline.py.

Builds OFAT (one-factor-at-a-time) sweeps that vary a single hyperparameter
relative to a per-cell baseline. Each generated row is one pipeline
invocation (subprocess-call to ``pipeline/run_pipeline.py``).

Watermark-yaml params (e.g. siren ``strength``, bitmark ``watermark_delta``)
are swapped via fresh watermark yamls minted under
``configs/watermarks/_ablation/`` — the planner reloads the watermark
config from disk per stage and at slug-computation time, so a top-level
``-o watermark.params.X=Y`` override is silently dropped. ``mint_watermark_yaml``
materializes a patched copy and the row points ``run.watermark_config`` at it.

Top-level entry points:
    Cell                — one (m1, mi, watermark, fraction) baseline.
    CellDefaults        — baseline finetuning hyperparameters.
    OFATAxes            — non-default values for each varied axis.
    plan_rows           — turn (cell, axes) into Row objects.
    submit_rows         — subprocess-call run_pipeline.py for each row.
    mint_watermark_yaml — temp-watermark-yaml helper.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION_WM_DIR = PROJECT_ROOT / "configs" / "watermarks" / "_ablation"


def mint_watermark_yaml(
    base_yaml: str | Path,
    key: str,
    value: float | int | str,
    out_dir: str | Path = DEFAULT_ABLATION_WM_DIR,
) -> Path:
    """Write a copy of ``base_yaml`` with ``key`` set to ``value``.

    ``key`` is a dot path inside the watermark config (e.g.
    ``watermark.params.strength``). The output filename encodes the
    leaf key and value so siblings stay distinguishable. Idempotent —
    re-minting a yaml that already matches on disk is a no-op.
    """
    base_path = Path(base_yaml)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    leaf = key.rsplit(".", 1)[-1]
    target = out_dir / f"{base_path.stem}_{leaf}{value}.yaml"

    cfg = OmegaConf.load(str(base_path))
    OmegaConf.update(cfg, key, value, merge=True)
    rendered = OmegaConf.to_yaml(cfg)

    if target.exists() and target.read_text() == rendered:
        return target

    target.write_text(rendered)
    return target


@dataclass(frozen=True)
class CellDefaults:
    """Baseline finetuning hyperparameters for a cell."""
    epochs: int
    learning_rate: float
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1


@dataclass(frozen=True)
class Cell:
    """A single (m1, mi, watermark, fraction) ablation baseline."""
    watermark_name: str            # short label, e.g. "siren"
    watermark_yaml: str            # path to base watermark yaml
    m1_model: str
    mi_model: str
    finetuning_yaml: str           # path to finetuning base yaml
    watermark_fraction: float
    defaults: CellDefaults
    strength_key: str              # dot path inside watermark yaml
    strength_default: float        # baseline value of strength_key


@dataclass(frozen=True)
class OFATAxes:
    """Non-default values per axis. The cell's defaults are always implicit.

    ``effective_batch_sizes`` entries are ``(eff_bs, train_batch_size,
    gradient_accumulation_steps)`` triples — explicit because not every
    eff_bs is reachable while keeping ``train_batch_size`` fixed.
    """
    learning_rates: Sequence[float] = ()
    epochs: Sequence[int] = ()
    effective_batch_sizes: Sequence[tuple[int, int, int]] = ()
    strengths: Sequence[float] = ()


@dataclass(frozen=True)
class Row:
    """One pipeline submission: a run name and the -o overrides to apply."""
    name: str
    overrides: list[tuple[str, str]] = field(default_factory=list)

    def as_args(self) -> list[str]:
        out: list[str] = []
        for k, v in self.overrides:
            out.extend(["-o", f"{k}={v}"])
        return out


def _ft(key: str) -> str:
    """Override-key prefix for finetuning chain_overrides."""
    return f"stages.finetuning.chain_overrides.{key}"


def _format_lr(lr: float) -> str:
    """Match the LR encoding used by ``src.paths.build_run_id``.

    e.g. 6e-4 -> '6e-4', 1e-5 -> '1e-5'. The same string is used both
    in the override value (where ``_coerce_override`` parses it back to
    a float) and in the run name (so the name lines up with the on-disk
    run_id).
    """
    return re.sub(r"e([+-])0+(\d)", r"e\1\2", f"{lr:.0e}")


def _baseline_overrides(cell: Cell, watermark_yaml: str) -> dict[str, str]:
    """Per-row constants: cell identity, fixed knobs, end_i."""
    profile = "high_vram" if cell.mi_model == "infinity_2b" else "default"
    return {
        "run.watermark_config": watermark_yaml,
        "run.m1_model_type": cell.m1_model,
        "run.mi_model_type": cell.mi_model,
        "run.finetuning_config": cell.finetuning_yaml,
        "run.chain.end_i": "1",
        "stages.finetuning.gpu_profile": profile,
        _ft("training.watermark_fraction"): str(cell.watermark_fraction),
        _ft("datasets.finetune_clean_source"): "generated",
    }


def _ft_defaults(d: CellDefaults) -> dict[str, str]:
    """Default finetuning overrides.

    grad_acc is set on BOTH ``training.*`` and ``training_setup.*`` because the
    planner reads it from ``training`` (for run_id), Infinity reads it from
    ``training_setup`` (actual training), and SD reads ``training_setup`` first
    then lets ``training`` overwrite. Setting both keeps run_id and the runner
    in agreement regardless of backend.
    """
    return {
        _ft("training.num_train_epochs"): str(d.epochs),
        _ft("training.learning_rate"): _format_lr(d.learning_rate),
        _ft("training.train_batch_size"): str(d.train_batch_size),
        _ft("training.gradient_accumulation_steps"): str(d.gradient_accumulation_steps),
        _ft("training_setup.gradient_accumulation_steps"): str(d.gradient_accumulation_steps),
    }


def _row_name(cell: Cell, axis_label: str, value: object) -> str:
    return (
        f"ablation_{cell.watermark_name}"
        f"_m1-{cell.m1_model}_mi-{cell.mi_model}"
        f"_f{cell.watermark_fraction:g}_{axis_label}{value}"
    )


def plan_rows(cell: Cell, axes: OFATAxes) -> list[Row]:
    """Build OFAT rows for one cell, varying one parameter at a time."""
    rows: list[Row] = []
    base = _baseline_overrides(cell, cell.watermark_yaml)
    defaults = _ft_defaults(cell.defaults)
    starting = {**base, **defaults}

    def emit(label: str, value: object, patch: dict[str, str], wm_yaml: str | None = None) -> None:
        ov = {**starting, **patch}
        if wm_yaml is not None:
            ov["run.watermark_config"] = wm_yaml
        rows.append(Row(_row_name(cell, label, value), list(ov.items())))

    for lr in axes.learning_rates:
        lr_str = _format_lr(lr)
        emit("lr", lr_str, {_ft("training.learning_rate"): lr_str})

    for eff_bs, bs, ga in axes.effective_batch_sizes:
        emit("eff_bs", eff_bs, {
            _ft("training.train_batch_size"): str(bs),
            _ft("training.gradient_accumulation_steps"): str(ga),
            _ft("training_setup.gradient_accumulation_steps"): str(ga),
        })

    for ep in axes.epochs:
        emit("ep", ep, {_ft("training.num_train_epochs"): str(ep)})

    leaf = cell.strength_key.rsplit(".", 1)[-1]
    for s in axes.strengths:
        wm_yaml = mint_watermark_yaml(cell.watermark_yaml, cell.strength_key, s)
        emit(leaf, s, {}, wm_yaml=str(wm_yaml))

    return rows


def runs_with_active_jobs(user: str) -> set[str]:
    """Return the set of ``run.name``s that currently have RUNNING or PENDING
    slurm jobs for ``user``. Used by :func:`submit_rows` to skip re-submitting
    rows that are already in flight.
    """
    if not user:
        return set()
    try:
        out = subprocess.check_output(
            ["scontrol", "show", "job", "-o"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    pattern = re.compile(r"/slurm_out/(?P<run>[^/]+)/")
    active: set[str] = set()
    for line in out.splitlines():
        if f"UserId={user}" not in line:
            continue
        state_match = re.search(r"JobState=(\S+)", line)
        std_match = re.search(r"StdOut=(\S+)", line)
        if not (state_match and std_match):
            continue
        if state_match.group(1) not in {"RUNNING", "PENDING"}:
            continue
        m = pattern.search(std_match.group(1))
        if m:
            active.add(m.group("run"))
    return active


def submit_rows(
    rows: Iterable[Row],
    config_path: str = "configs/pipeline/run_pipeline.yaml",
    cluster: str = "",
    extra_args: Sequence[str] = (),
    python: str | None = None,
) -> None:
    """Subprocess-call ``pipeline/run_pipeline.py`` once per row.

    Each call submits its own SLURM DAG. Filesystem state (M1 clean,
    M1 watermarked when the slug is unchanged) is shared between rows
    via the planner's existing completion checks. Rows whose ``run.name``
    already has RUNNING/PENDING slurm jobs are skipped — re-running the
    sweep won't duplicate in-flight pipelines.
    """
    python = python or sys.executable
    rows = list(rows)
    user = os.environ.get("USER", "")
    active = runs_with_active_jobs(user)
    print("==================== Ablation sweep ====================")
    print(f"Cluster:         {cluster}")
    print(f"Config:          {config_path}")
    print(f"Pending rows:    {len(rows)}")
    print(f"Active in slurm: {len(active)} (will be skipped)")
    print("========================================================\n", flush=True)
    submitted = 0
    skipped = 0
    for idx, row in enumerate(rows, 1):
        if row.name in active:
            print(f">>> [{idx}/{len(rows)}] {row.name} — SKIP (active in slurm)", flush=True)
            skipped += 1
            continue
        print(f">>> [{idx}/{len(rows)}] {row.name}", flush=True)
        cmd = [
            python, "pipeline/run_pipeline.py",
            "--config_path", config_path,
            "--cluster", cluster,
            "-o", f"run.name={row.name}",
            *row.as_args(),
            *extra_args,
        ]
        subprocess.run(cmd, check=True)
        submitted += 1
        print(flush=True)
    print(f"Submitted {submitted} pipeline run(s); skipped {skipped} already in flight.")
