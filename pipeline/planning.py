"""Planning-time stage completeness checks.

The orchestrator (``pipeline.run_pipeline``) consults this module to decide
which stages can be skipped because their artifacts already exist. It runs
on the login node where the full GPU stack (flash-attn, diffusers, etc.)
may not be importable, so this module — and everything it pulls in — must
stay free of those heavy dependencies.

That is why heavy imports inside ``pipeline.stages.*`` and
``pipeline.eval_common`` were moved from module top into the functions
that actually need them. The resulting stage modules are import-safe at
planning time; this module dispatches their lightweight ``is_*_complete``
helpers by stage name.
"""

from __future__ import annotations

from typing import Callable

from pipeline.stage import extract_arg


def _stage_kind(stage: str) -> str:
    """Map an orchestrator stage name (e.g. 'detection_mi_supervised') to its
    underlying kind ('generation' / 'detection' / 'finetuning' / 'evaluation').
    """
    if stage.startswith("generation"):
        return "generation"
    if stage.startswith("detection"):
        return "detection"
    if stage == "finetuning":
        return "finetuning"
    if stage == "evaluation":
        return "evaluation"
    raise ValueError(f"Unknown stage '{stage}'")


def _load_stage_config(kind: str, config_path: str, watermark_config_path: str | None):
    """Load a stage config using the same merge semantics the stage itself uses,
    but without importing the heavy execution-time modules."""
    if kind == "finetuning":
        from src.finetuning.runner import load_finetune_config
        return load_finetune_config(config_path, watermark_config_path=watermark_config_path)
    from src.utils.commonargs import load_config
    return load_config(config_path, watermark_config_path=watermark_config_path)


def _check_generation(config, command: list[str]) -> bool:
    from pipeline.stages.generation import is_generation_complete
    return is_generation_complete(config)


def _check_detection(config, command: list[str]) -> bool:
    from pipeline.stages.detection import is_detection_complete, resolve_detection_paths
    output_dir = extract_arg(command, "--output_dir") or resolve_detection_paths(config).get("output_dir")
    if not output_dir:
        return False
    return is_detection_complete(config, output_dir)


def _check_finetuning(config, command: list[str]) -> bool:
    from pipeline.stages.finetuning import is_finetuning_complete
    return is_finetuning_complete(config)


def _check_evaluation(config, command: list[str]) -> bool:
    from pipeline.stages.evaluation import is_evaluation_complete, resolve_evaluation_paths
    output_dir = extract_arg(command, "--output_dir") or resolve_evaluation_paths(config).get("output_dir")
    if not output_dir:
        return False
    return is_evaluation_complete(config, output_dir)


_CHECKS: dict[str, Callable] = {
    "generation": _check_generation,
    "detection": _check_detection,
    "finetuning": _check_finetuning,
    "evaluation": _check_evaluation,
}


def is_stage_complete(
    stage: str,
    config_path: str,
    watermark_config_path: str | None,
    command: list[str],
) -> bool:
    """Return True iff the stage's artifacts exist and cover the configured work.

    Exceptions are not caught here: the orchestrator is responsible for
    logging them so import or path-resolution failures surface during
    planning instead of being silently demoted to "missing artifacts".
    """
    kind = _stage_kind(stage)
    config = _load_stage_config(kind, config_path, watermark_config_path)
    return _CHECKS[kind](config, command)
