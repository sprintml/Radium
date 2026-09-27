"""Top-level pipeline planner/orchestrator.

Supports two planning modes:
1. Single-run mode (one generation -> finetuning -> evaluation path)
2. Chain mode over ``i`` (for M1 -> M2 -> ... -> Mi), configured via
   ``run.chain.start_i`` and ``run.chain.end_i``.

Chain mode has two sub-modes:
- Manual (default): ``command_template`` tokens use ``{i}`` to reference
  pre-existing per-iteration config files.
- Auto (``run.auto_generate_configs: true``): each stage declares a
  ``base_config`` + ``chain_overrides`` (applied every iteration) and
  optional ``sweeps`` (per-parameter expansion applied to specific chains).
  The pipeline writes generated configs to ``run.generated_configs_dir``
  and appends ``--config_path`` to the stage command automatically.

For each stage, completion is checked by importing the stage's own
``is_*_complete`` function and calling it with that stage's config.
Stages whose outputs already exist are skipped; others run in order.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
import os
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from pipeline.stage import StageOrchestrator, extract_arg
from src.utils.commonargs import cfg_get as _cfg_get, load_config

# Map stage name -> module path. Add one entry here to register a new stage.
_STAGE_MODULES: dict[str, str] = {
    "generation_mi": "pipeline.stages.generation",
    "generation_unsupervised_mi": "pipeline.stages.generation",
    "generation_m1_clean": "pipeline.stages.generation",
    "generation_m1_clean_unsupervised": "pipeline.stages.generation",
    "generation_m1_watermarked": "pipeline.stages.generation",
    "finetuning": "pipeline.stages.finetuning",
    "detection_m1_supervised": "pipeline.stages.detection",
    "detection_m1_unsupervised": "pipeline.stages.detection",
    "detection_m1_watermarked": "pipeline.stages.detection",
    "detection_mi_supervised": "pipeline.stages.detection",
    "detection_mi_unsupervised": "pipeline.stages.detection",
    "evaluation": "pipeline.stages.evaluation",
    "attack": "pipeline.stages.attack",
}


def _get_stage(name: str) -> StageOrchestrator:
    module_path = _STAGE_MODULES.get(name)
    if not module_path:
        raise ValueError(f"Unknown stage '{name}'. Register it in _STAGE_MODULES.")
    import importlib
    return importlib.import_module(module_path).STAGE


def _default_command(stage: str) -> list[str]:
    module = _STAGE_MODULES.get(stage)
    if not module:
        raise ValueError(f"No default command for unknown stage '{stage}'")
    return ["python", "-m", module]


def _format_context(i: int) -> dict[str, int]:
    return {"i": i, "next_i": i + 1, "prev_i": i - 1}


def _resolve_per_i(value, i: int):
    """Recursively resolve any {_per_i: {i: value, ...}} markers to the scalar
    for this iteration. Markers may appear at any depth (CLI -o overrides
    nest dotted keys, so the marker can land deep inside the value tree)."""
    if isinstance(value, dict):
        if "_per_i" in value:
            per_i = value["_per_i"]
            if not isinstance(per_i, dict):
                raise ValueError(
                    f"_per_i must be a mapping of iteration -> value, got {type(per_i).__name__}"
                )
            keyed = {int(k): v for k, v in per_i.items()}
            if i not in keyed:
                raise ValueError(
                    f"_per_i override missing entry for i={i}; provided iterations: {sorted(keyed)}"
                )
            return _resolve_per_i(keyed[i], i)
        return {k: _resolve_per_i(v, i) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_per_i(v, i) for v in value]
    return value


def _format_token(value, i: int):
    if isinstance(value, str):
        return value.format(**_format_context(i))
    if OmegaConf.is_config(value):
        try:
            value = OmegaConf.to_container(value, resolve=True)
        except Exception:
            pass
    return _resolve_per_i(value, i)


def _apply_overrides(cfg, overrides: dict) -> None:
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=True)


def _validate_watermark_generation_compatibility(
    watermark_config_path: str | None, m1_model_type: str | None
) -> None:
    """Hard-fail early if the watermark can't run on the requested M1 architecture.

    In-generation watermarks must declare ``watermark.generation_compatible_architectures``
    (a non-empty list of M1 model types they can embed into); the planner asserts
    that ``run.m1_model_type`` is in that list. Post-generation watermarks default
    to all architectures — the field may be omitted.
    """
    if not watermark_config_path:
        return
    wm_cfg = OmegaConf.load(watermark_config_path)
    wm = wm_cfg.get("watermark") if hasattr(wm_cfg, "get") else None
    if not wm:
        return
    wm_type = wm.get("type")
    method = wm.get("method")
    compat = wm.get("generation_compatible_architectures")
    if wm_type == "in_generation" and not compat:
        raise ValueError(
            f"Watermark {method!r} ({watermark_config_path}) has type=in_generation "
            "but does not declare watermark.generation_compatible_architectures. "
            "List the M1 architectures it can embed into, e.g. "
            "`generation_compatible_architectures: [sd21]`."
        )
    if compat and m1_model_type and m1_model_type not in list(compat):
        raise ValueError(
            f"Watermark {method!r} ({watermark_config_path}) is not compatible with "
            f"run.m1_model_type={m1_model_type!r}; allowed: {list(compat)}."
        )


def _load_config_with_overrides(base_config_path: str, overrides: dict, watermark_config_path: str | None = None):
    cfg = load_config(base_config_path, watermark_config_path=watermark_config_path)
    _apply_overrides(cfg, overrides)
    # Overrides can introduce m1_model_type / mi_model_type that were absent at
    # load_config time; re-compose so the model-base YAML (which carries
    # paths.out_base and default model.*) gets merged underneath. Merge is
    # idempotent when the base is already present, so re-running is safe.
    from src.utils import compose_generation_config
    return compose_generation_config(cfg)


def _is_generation_stage(stage: str) -> bool:
    return _STAGE_MODULES.get(stage) == "pipeline.stages.generation"


def _is_detection_stage(stage: str) -> bool:
    return _STAGE_MODULES.get(stage) == "pipeline.stages.detection"


def _get_generation_target(stage: str) -> str:
    if "m1_watermarked" in stage:
        return "m1_watermarked"
    if "m1_clean" in stage:
        return "m1_clean"
    return "mi"


def _get_detection_target(stage: str) -> str:
    """'m1_watermarked' for detection_m1_watermarked, 'm1_clean' for other
    detection_m1_*, 'mi' for detection_mi_*."""
    if "m1_watermarked" in stage:
        return "m1_watermarked"
    return "m1_clean" if "_m1_" in stage else "mi"


def _stage_targets_mi(stage: str) -> bool:
    """True if the stage consumes/produces Mi artifacts (and therefore needs
    mi_model_type + Mi checkpoint path). False for M1-only stages."""
    if _is_generation_stage(stage):
        return _get_generation_target(stage) == "mi"
    if _is_detection_stage(stage):
        return _get_detection_target(stage) == "mi"
    # finetuning & evaluation always touch both sides.
    return True


def _stage_enabled_for_iteration(spec: dict, i: int) -> bool:
    only_iterations = spec.get("only_iterations")
    if only_iterations is None:
        return True
    if isinstance(only_iterations, (list, tuple, set)) or OmegaConf.is_list(only_iterations):
        return i in {int(v) for v in only_iterations}
    return i == int(only_iterations)




@dataclass
class PlannedStep:
    name: str
    status: str
    reason: str
    command: list[str] | None = None
    i: int | None = None
    depends_on: list[str] = field(default_factory=list)
    # Optional name of a GPU profile defined under `gpu_profiles:` in the
    # cluster config. None -> the cluster's default profile.
    gpu_profile: str | None = None


def _is_stage_complete(stage: str, command: list[str]) -> bool:
    config_path = extract_arg(command, "--config_path")
    if not config_path:
        return False
    watermark_config_path = extract_arg(command, "--watermark_config")
    from pipeline.planning import is_stage_complete
    try:
        return is_stage_complete(stage, config_path, watermark_config_path, command)
    except Exception as e:
        print(f"[plan] {stage}: completeness check failed ({type(e).__name__}: {e})", file=sys.stderr)
        return False


def _format_command(command: list[str], i: int) -> list[str]:
    formatted: list[str] = []
    for token in command:
        if isinstance(token, str):
            formatted.append(_format_token(token, i))
        else:
            formatted.append(str(token))
    return formatted


def _plan_step(
    stage: str,
    command: list[str] | None,
    i: int | None = None,
    gpu_profile: str | None = None,
) -> PlannedStep:
    done = _is_stage_complete(stage, command) if command else False
    suffix = f"_M{i}" if i is not None else ""
    return PlannedStep(
        name=f"{stage}{suffix}",
        status="skip" if done else "pending",
        reason="artifacts exist" if done else "missing artifacts",
        command=command,
        i=i,
        gpu_profile=gpu_profile,
    )


def _resolve_yaml_depends_on(raw_deps: list, i: int, step_name_set: set[str]) -> list[str]:
    """Format dep templates with {i}/{prev_i}/{next_i} and filter to steps that exist in the plan."""
    ctx = _format_context(i)
    resolved = []
    for dep in raw_deps:
        if isinstance(dep, str):
            name = dep.format(**ctx)
            if name in step_name_set:
                resolved.append(name)
    return resolved


def _assign_dependencies(steps: list[PlannedStep], yaml_deps: dict[str, list] | None = None) -> None:
    """Assign depends_on to every step.

    Default: no dependencies (all steps run in parallel).
    A stage with depends_on in the YAML config overrides this — the list is
    formatted with {i}/{prev_i}/{next_i} and filtered to steps present in the plan.
    """
    names = {s.name for s in steps}
    for step in steps:
        if yaml_deps and step.name in yaml_deps:
            step.depends_on = _resolve_yaml_depends_on(yaml_deps[step.name], step.i or 0, names)
        else:
            step.depends_on = []


# ---------------------------------------------------------------------------
# Manual chain plan (pre-existing per-i config files via command_template)
# ---------------------------------------------------------------------------

def _build_single_plan(config) -> list[PlannedStep]:
    stage_order = (
        _cfg_get(config, "run.chain.stages")
        or list((_cfg_get(config, "stages") or {}).keys())
        or ["generation", "detection", "evaluation", "finetuning"]
    )
    steps: list[PlannedStep] = []
    for stage in stage_order:
        spec = dict(_cfg_get(config, f"stages.{stage}", {}) or {})
        command = spec.get("command")
        steps.append(_plan_step(stage, command, i=None, gpu_profile=spec.get("gpu_profile")))
    return steps


def _build_chain_plan(config) -> list[PlannedStep]:
    steps: list[PlannedStep] = []
    chain_cfg = dict(_cfg_get(config, "run.chain", {}) or {})
    start_i = int(chain_cfg.get("start_i", 1))
    end_i = int(chain_cfg.get("end_i", start_i))
    if end_i < start_i:
        raise ValueError(f"run.chain.end_i must be >= start_i (got {end_i} < {start_i})")

    stage_order = chain_cfg.get("stages", ["generation", "detection", "evaluation", "finetuning"])
    yaml_deps: dict[str, list] = {}
    for i in range(start_i, end_i + 1):
        for stage in stage_order:
            spec = dict(_cfg_get(config, f"stages.{stage}", {}) or {})
            if not _stage_enabled_for_iteration(spec, i):
                continue
            cmd_template = spec.get("command_template", spec.get("command"))
            if cmd_template is None:
                continue
            command = _format_command(cmd_template, i=i)
            steps.append(_plan_step(stage, command, i=i, gpu_profile=spec.get("gpu_profile")))
            if "depends_on" in spec:
                yaml_deps[f"{stage}_M{i}"] = list(spec["depends_on"])
    _assign_dependencies(steps, yaml_deps or None)
    return steps



def _coerce_override(value: str) -> bool | int | float | str | list:
    """Cast a substituted string override back to the most specific scalar type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [_coerce_override(item.strip()) for item in inner.split(",")]
    return value


def _write_generated_config(
    base_config_path: str,
    overrides: dict,
    out_path: Path,
    watermark_config_path: str | None = None,
) -> None:
    cfg = _load_config_with_overrides(base_config_path, overrides, watermark_config_path=watermark_config_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(out_path))


def _derive_generation_output_dir(
    generation_cfg,
    *,
    generation_target: str,
    run_id: str | None,
    model_iteration: int,
) -> str | None:
    from src.paths import (
        ProjectPaths,
        prompt_set_name,
        resolve_m1_model_type,
        resolve_mi_model_type,
        resolve_wm_params,
    )

    try:
        layout = ProjectPaths.from_config(generation_cfg)
    except ValueError:
        return None

    wm_cfg = generation_cfg.get("watermark", {}) or {}
    method = wm_cfg.get("method")
    wm_params = resolve_wm_params(generation_cfg)
    m1_model_type = resolve_m1_model_type(generation_cfg)
    # Mi may differ from M1 in cross-architecture chains; fall back to M1 only
    # when Mi is not configured (single-architecture chain).
    mi_model_type = resolve_mi_model_type(generation_cfg) or m1_model_type

    datasets_cfg = generation_cfg.get("datasets", {}) or {}
    dataset = datasets_cfg.get("name") or "coco"
    prompt_iteration = int(wm_cfg.get("iteration", model_iteration) or model_iteration)
    set_name = datasets_cfg.get("set_name") or prompt_set_name(prompt_iteration)

    if not m1_model_type:
        return None

    if generation_target == "m1_clean":
        return str(layout.clean_generated(m1_model_type, dataset, set_name))

    if generation_target == "m1_watermarked":
        if not all([method, wm_params]):
            return None
        return str(layout.m1_generated(m1_model_type, method, wm_params, dataset, set_name))

    if generation_target != "mi":
        raise ValueError(f"Unsupported generation_target: {generation_target}")

    if not all([run_id, method, wm_params, mi_model_type]):
        return None

    return str(
        layout.mi_generated(
            m1_model_type, method, wm_params, run_id, model_iteration, mi_model_type, dataset, set_name
        )
    )


def _derive_mi_model_path(
    finetuning_spec: dict,
    i: int,
    watermark_config: str | None = None,
    paths_out_base: str | None = None,
    base_config_path: str | None = None,
    m1_model_type: str | None = None,
    mi_model_type: str | None = None,
) -> str | None:
    """Compute the filesystem path of the Mi trained model.

    Used to inject model.pretrained_model_name_or_path into the generation config
    for iteration i, so generation uses Mi's weights rather than M1's. The
    watermark config is merged in first so watermark.method / params are
    visible for the path layout.
    """
    from src.paths import ProjectPaths, resolve_m1_model_type, resolve_mi_model_type, resolve_wm_params

    run_id = _derive_run_id(finetuning_spec, i, base_config_path=base_config_path)
    if not run_id:
        return None

    base_config = base_config_path or finetuning_spec.get("base_config")
    if not base_config:
        return None

    try:
        cfg = OmegaConf.load(base_config)
        OmegaConf.set_struct(cfg, False)
        if m1_model_type:
            cfg["m1_model_type"] = m1_model_type
        if mi_model_type:
            cfg["mi_model_type"] = mi_model_type
        if watermark_config:
            cfg = OmegaConf.merge(OmegaConf.load(watermark_config), cfg)
        for k, v in (dict(finetuning_spec.get("chain_overrides") or {})).items():
            formatted = _format_token(v, i)
            if isinstance(formatted, str):
                formatted = _coerce_override(formatted)
            OmegaConf.update(cfg, k, formatted, merge=True)
        if paths_out_base:
            OmegaConf.update(cfg, "paths.out_base", paths_out_base, merge=True)
        layout = ProjectPaths.from_config(cfg)
        wm = cfg.get("watermark", {}) or {}
        model_iteration = int(wm.get("iteration", i))
        method = wm.get("method")
        wm_params = resolve_wm_params(cfg)
        m1_resolved = resolve_m1_model_type(cfg)
        mi_resolved = resolve_mi_model_type(cfg)
        if not all([method, wm_params, m1_resolved, mi_resolved]):
            return None
        return str(layout.mi_model_dir(m1_resolved, method, wm_params, run_id, model_iteration, mi_resolved))
    except Exception:
        return None


def _derive_run_id(finetuning_spec: dict, i: int, base_config_path: str | None = None) -> str | None:
    """Compute run_id from the finetuning base_config + chain_overrides for iteration i.

    Uses src.finetuning.runner.load_finetune_config so the merge order
    matches what finetuning actually trains with (the backend-appropriate
    base from configs/finetuning/ underneath base_config). Reads training
    params strictly — missing keys raise.
    """
    from src.paths import build_run_id
    from src.finetuning.runner import load_finetune_config

    base_config = base_config_path or finetuning_spec.get("base_config")
    if not base_config:
        return None
    cfg = load_finetune_config(base_config)
    for k, v in (dict(finetuning_spec.get("chain_overrides") or {})).items():
        formatted = _format_token(v, i)
        if isinstance(formatted, str):
            formatted = _coerce_override(formatted)
        OmegaConf.update(cfg, k, formatted, merge=True)
    training = cfg.get("training", {}) or {}
    missing = [k for k in ("watermark_fraction", "num_train_epochs", "learning_rate", "train_batch_size") if training.get(k) is None]
    if missing:
        raise ValueError(
            f"Cannot derive run_id from {base_config}: training is missing required keys "
            f"{missing}. Set them in the finetuning config (or its merged base under configs/finetuning/)."
        )
    eff_bs = int(training["train_batch_size"]) * int(training.get("gradient_accumulation_steps", 1) or 1)
    clean_source = (cfg.get("datasets", {}) or {}).get("finetune_clean_source", "generated")
    # Optional suffix lets chains at the same fraction but seeded from
    # different source models land in disjoint run dirs (cleani2 collision
    # protection — see run_pipeline.sh's --clean-i2 branch).
    suffix = training.get("run_id_suffix")
    return build_run_id(
        watermark_fraction=float(training["watermark_fraction"]),
        epochs=int(training["num_train_epochs"]),
        learning_rate=float(training["learning_rate"]),
        effective_batch_size=eff_bs,
        clean_source=clean_source,
        suffix=str(suffix) if suffix else None,
    )


def _read_watermark_type(watermark_config_path: str | None) -> str | None:
    """Peek at a watermark config to learn its type (in_generation / post_generation)."""
    if not watermark_config_path:
        return None
    try:
        cfg = OmegaConf.load(watermark_config_path)
        return cfg.get("watermark", {}).get("type")
    except Exception:
        return None


def _derive_m1_clean_dir(stage_cfg, iteration: int) -> str | None:
    """Derive the M1 clean prompt_set_{iteration} directory from the merged config."""
    from src.paths import ProjectPaths, prompt_set_name, resolve_m1_model_type
    m1 = resolve_m1_model_type(stage_cfg)
    dataset = (stage_cfg.get("datasets", {}) or {}).get("name") or "coco"
    if not m1:
        return None
    try:
        layout = ProjectPaths.from_config(stage_cfg)
    except ValueError:
        return None
    return str(layout.clean_generated(m1, dataset, prompt_set_name(iteration)))


def _resolve_generated_configs_dir(config) -> Path:
    raw = Path(str(_cfg_get(config, "run.generated_configs_dir", "pipeline_configs")))
    if raw.is_absolute():
        base = raw
    else:
        out_base = _cfg_get(config, "run.paths_out_base") or os.environ.get("RADIOACTIVITY_OUT")
        if not out_base:
            raise ValueError(
                "Relative run.generated_configs_dir requires run.paths_out_base "
                "or the RADIOACTIVITY_OUT environment variable."
            )
        base = Path(str(out_base)) / raw
    # Namespace by run.name so concurrent sweep variants don't overwrite each
    # other's per-stage YAMLs (the slurm command captures the path, not the
    # file content, so a shared dir silently makes every variant read whichever
    # variant's config was written last).
    run_name = _cfg_get(config, "run.name")
    if run_name:
        base = base / str(run_name)
    return base


def _build_auto_chain_plan(config) -> list[PlannedStep]:
    chain_cfg = dict(_cfg_get(config, "run.chain", {}) or {})
    start_i = int(chain_cfg.get("start_i", 1))
    end_i = int(chain_cfg.get("end_i", start_i))
    if end_i < start_i:
        raise ValueError(f"run.chain.end_i must be >= start_i (got {end_i} < {start_i})")
    stage_order = chain_cfg.get("stages", ["generation", "detection", "evaluation", "finetuning"])
    configs_dir = _resolve_generated_configs_dir(config)

    steps: list[PlannedStep] = []
    yaml_deps: dict[str, list] = {}
    run_watermark_config = _cfg_get(config, "run.watermark_config")
    _validate_watermark_generation_compatibility(
        run_watermark_config, _cfg_get(config, "run.m1_model_type")
    )
    # Optional: explicit project output path. When set, injected into every
    # stage's overrides as paths.out_base so generated artifacts do not depend
    # on the RADIOACTIVITY_OUT environment variable.
    run_paths_out_base = _cfg_get(config, "run.paths_out_base")
    for i in range(start_i, end_i + 1):
        ft_spec = dict(_cfg_get(config, "stages.finetuning", {}) or {})
        ft_watermark_config = ft_spec.get("watermark_config") or run_watermark_config
        ft_base_config = ft_spec.get("base_config") or _cfg_get(config, "run.finetuning_config")
        m1_model_type = _cfg_get(config, "run.m1_model_type")
        mi_model_type = _cfg_get(config, "run.mi_model_type") or m1_model_type
        if not m1_model_type:
            raise ValueError(
                "run.m1_model_type is required (set it in the pipeline config or via "
                "-o run.m1_model_type=<type>). Same applies to run.mi_model_type."
            )
        derived_run_id = _derive_run_id(ft_spec, i, base_config_path=ft_base_config)
        mi_model_path = _derive_mi_model_path(
            ft_spec,
            i,
            watermark_config=ft_watermark_config,
            paths_out_base=run_paths_out_base,
            base_config_path=ft_base_config,
            m1_model_type=m1_model_type,
            mi_model_type=mi_model_type,
        )
        # For chain step i ≥ 2, finetuning at this step starts from M{i}
        # (the model that step i-1's finetuning wrote). At i=1 we keep the
        # base model from configs/models/<mi_model_type>.yaml.
        prev_mi_model_path = None
        if i > 1:
            prev_mi_model_path = _derive_mi_model_path(
                ft_spec,
                i - 1,
                watermark_config=ft_watermark_config,
                paths_out_base=run_paths_out_base,
                base_config_path=ft_base_config,
                m1_model_type=m1_model_type,
                mi_model_type=mi_model_type,
            )

        for stage in stage_order:
            spec = dict(_cfg_get(config, f"stages.{stage}", {}) or {})
            if not _stage_enabled_for_iteration(spec, i):
                continue
            base_config = spec.get("base_config") or (
                _cfg_get(config, "run.generation_config") if _is_generation_stage(stage) else
                _cfg_get(config, "run.detection_config") if _is_detection_stage(stage) else
                _cfg_get(config, "run.finetuning_config") if stage == "finetuning" else
                _cfg_get(config, "run.evaluation_config") if stage == "evaluation" else
                None
            )
            if not base_config:
                continue
            # Per-stage watermark_config overrides run.watermark_config.
            watermark_config = spec.get("watermark_config") or run_watermark_config

            overrides: dict = {}
            for k, v in (dict(spec.get("chain_overrides") or {})).items():
                formatted = _format_token(v, i)
                overrides[k] = _coerce_override(formatted) if isinstance(formatted, str) else formatted

            # Inject m1/mi_model_type only when the stage actually consumes them,
            # so compose_generation_config picks the correct base-model YAML:
            #   - M1-only stages (generation_m1_*, detection_m1_*): m1 only.
            #   - Mi stages: both (m1 for path context, mi for the active model).
            #   - finetuning/evaluation: both.
            stage_mi = _stage_targets_mi(stage)
            if m1_model_type:
                overrides.setdefault("m1_model_type", m1_model_type)
            if mi_model_type and stage_mi:
                overrides.setdefault("mi_model_type", mi_model_type)

            if run_paths_out_base:
                overrides.setdefault("paths.out_base", run_paths_out_base)

            # Auto-inject watermark.run_id for detection/evaluation unless
            # already provided explicitly in chain_overrides.
            if (_is_detection_stage(stage) or stage == "evaluation") and derived_run_id is not None:
                overrides.setdefault("watermark.run_id", derived_run_id)

            # Auto-inject the Mi checkpoint path for any stage that needs to
            # load the Mi model — generation_mi (to generate from Mi) and
            # detection_mi_* (to invert/detect against Mi-generated images
            # using the Mi pipeline, e.g. TreeRing's DDIM).
            if stage_mi and mi_model_path is not None and (
                _is_generation_stage(stage) or _is_detection_stage(stage)
            ):
                overrides["model.model_path"] = mi_model_path

            # Chain finetuning: at chain step i ≥ 2, Mi finetuning starts
            # from M{i} (the prior step's output) instead of the base model
            # in configs/models/<mi_model_type>.yaml. SD reads model.model_path
            # for `pretrained_model_name_or_path`; Infinity reads
            # model.pretrained_model_name_or_path for `--rush_resume`. Override
            # both so the chain is honored regardless of backend.
            if stage == "finetuning" and prev_mi_model_path is not None:
                overrides.setdefault("model.model_path", prev_mi_model_path)
                overrides.setdefault("model.pretrained_model_name_or_path", prev_mi_model_path)

            # Auto-inject the generation output directory so each prompt slice
            # is written to a distinct Mi/{set_name} directory.
            if _is_generation_stage(stage) and base_config and (
                derived_run_id is not None or not stage_mi
            ):
                generation_cfg = _load_config_with_overrides(
                    base_config, overrides, watermark_config_path=watermark_config
                )
                derived_output_dir = _derive_generation_output_dir(
                    generation_cfg,
                    generation_target=_get_generation_target(stage),
                    run_id=derived_run_id,
                    model_iteration=i + 1,
                )
                if derived_output_dir is not None:
                    overrides["datasets.output_dir"] = derived_output_dir

                # Post-generation watermarks consume M1 clean images and
                # re-emit them watermarked. Point datasets.input_dir at
                # M1 clean prompt_set_{i}. The generation_m1_clean_M{i}
                # dep is injected below so this only runs after the
                # clean set exists.
                if (
                    _get_generation_target(stage) == "m1_watermarked"
                    and _read_watermark_type(watermark_config) == "post_generation"
                ):
                    clean_dir = _derive_m1_clean_dir(generation_cfg, iteration=i)
                    if clean_dir is not None:
                        overrides["datasets.input_dir"] = clean_dir
                    dep_name = f"generation_m1_clean_M{i}"
                    spec_deps = list(spec.get("depends_on") or [])
                    if dep_name not in spec_deps:
                        spec_deps.append(dep_name)
                        spec = dict(spec)
                        spec["depends_on"] = spec_deps

            cfg_path = configs_dir / f"{stage}_M{i}.yaml"
            _write_generated_config(base_config, overrides, cfg_path, watermark_config_path=watermark_config)

            cmd = [_format_token(t, i) if isinstance(t, str) else str(t) for t in (spec.get("command") or _default_command(stage))]
            cmd += ["--config_path", str(cfg_path)]
            steps.append(_plan_step(stage, cmd, i=i, gpu_profile=spec.get("gpu_profile")))
            if "depends_on" in spec:
                yaml_deps[f"{stage}_M{i}"] = list(spec["depends_on"])

    _assign_dependencies(steps, yaml_deps or None)
    return steps


def _build_plan(config) -> list[PlannedStep]:
    chain_cfg = dict(_cfg_get(config, "run.chain", {}) or {})
    if not (("start_i" in chain_cfg) or ("end_i" in chain_cfg)):
        return _build_single_plan(config)
    if _cfg_get(config, "run.auto_generate_configs", False):
        return _build_auto_chain_plan(config)
    return _build_chain_plan(config)


def _execute_steps(steps: list[PlannedStep], cluster: str | None = None, slurm_base_dir: Path | None = None) -> None:
    """Execute pending steps.

    If cluster is provided, submit all stages as sbatch jobs in one pass.
    Dependency job IDs are wired via --dependency=afterok so that independent
    jobs run in parallel while dependent ones queue automatically.
    Skipped steps (already complete) are excluded from dependency resolution —
    their dependents submit without waiting.
    If cluster is not provided, run commands locally in order.
    """
    from pipeline.slurm import submit_job

    submitted: dict[str, str] = {}  # step name -> SLURM job ID

    for step in steps:
        if step.status != "pending":
            continue
        if not step.command:
            raise ValueError(f"Step '{step.name}' is pending but has no command")

        if cluster:
            if slurm_base_dir is None:
                slurm_base_dir = Path(__file__).resolve().parents[1] / "slurm_base"
            dep_ids = [submitted[dep] for dep in step.depends_on if dep in submitted]
            job_id = submit_job(
                step.command,
                cluster,
                slurm_base_dir,
                dependency_job_ids=dep_ids or None,
                gpu_profile=step.gpu_profile,
            )
            submitted[step.name] = job_id
            profile_info = f" [gpu_profile={step.gpu_profile}]" if step.gpu_profile else ""
            dep_info = f" (after jobs {dep_ids})" if dep_ids else ""
            print(f"[{step.name}] Submitted sbatch job {job_id}{profile_info}{dep_info}")
        else:
            subprocess.run(step.command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-level pipeline planner/orchestrator")
    parser.add_argument("--config_path", required=True, help="Path to pipeline config yaml")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the execution plan, but do not run any stage commands.",
    )
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dot-path override applied to the loaded pipeline config (repeatable). "
        "Examples: -o run.watermark_config=configs/watermarks/bitmark.yaml "
        "-o run.m1_model_type=sd21 -o run.mi_model_type=infinity_2b "
        "-o run.finetuning_config=configs/finetuning/infinity/infinity_2b.yaml "
        "-o training.watermark_fraction=0.5",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    for item in args.override:
        if "=" not in item:
            raise ValueError(f"--override expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        OmegaConf.update(config, key.strip(), _coerce_override(value.strip()), merge=True)
    steps = _build_plan(config)

    for step in steps:
        dep_info = f" | deps: {step.depends_on}" if step.depends_on else ""
        print(f"[{step.status}] {step.name}: {step.reason}{dep_info}")

    if args.dry_run:
        pending = sum(1 for s in steps if s.status == "pending")
        print(f"\nDry run — {pending} pending step(s) would be executed. No commands launched.")
        return

    if args.cluster:
        print(f"\nSubmitting stages to cluster: {args.cluster}")
    else:
        print("\nRunning stages locally (no cluster specified)")

    _execute_steps(steps, cluster=args.cluster)


if __name__ == "__main__":
    main()
