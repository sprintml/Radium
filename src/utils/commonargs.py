import os
"""Common argument helpers and a small args dataclass.

Provide a light-weight programmatic argument container for pipelines and
components. This is intentionally simple (no CLI parsing) and supports loading
from dicts or environment variables.

Usage:
  from src.utils.args import CommonArgs, load_args_from_dict
  args = load_args_from_dict({"device": "cuda", "seed": 42})
  pipeline = GenerationPipeline(model, args=args)

"""

from pathlib import Path
from typing import Any

import random
import numpy as np
from omegaconf import OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _PROJECT_ROOT / "configs" / "models"


def cfg_get(config, path: str, default=None):
    obj = config
    for part in path.split("."):
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            try:
                obj = obj.get(part)
            except Exception:
                obj = getattr(obj, part, None)
    return default if obj is None else obj


def _load_yaml_if_exists(path: Path):
    if not path.exists():
        return None
    return OmegaConf.load(str(path))


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} must be set")
    return value


def _needs_model_base(config) -> bool:
    """Apply model-base merge to stages that care about the model:
    generation, detection, and finetuning. Excluded:
      - evaluation (no model loaded; reads cached score files)
    """
    if config is None or not hasattr(config, "get"):
        return False
    if config.get("watermark") is None and config.get("training") is None:
        return False
    if config.get("evaluation_scenarios") is not None:
        return False
    return True


def compose_generation_config(config):
    """Merge model defaults from configs/models/<model_type>.yaml under the config.

    Merge precedence (lowest to highest priority):
      M1 model file  ->  primary (mi or M1) model file  ->  concrete config

    Model-type selection:
      - primary = mi_model_type if set (Mi stages: Mi is the model
        generating/inverting), else m1_model_type.
    Explicit values in the concrete config always win over the merged base.
    Applies to generation, detection, and finetuning — all need model.*; the
    generation/detection stages additionally read dataset_params sampling
    defaults, and finetuning reads the finetuning.* subtree.

    Cross-architecture support: when ``m1_model_type`` differs from
    ``mi_model_type``, the M1 model file is layered *under* the primary
    (mi) file so M1-only sidecars survive (e.g. the ``infinity:`` block
    from infinity_2b.yaml that bitmark's detector reads). The M1 ``model:``
    block is additionally promoted to ``m1_model:`` so cross-arch watermark
    detectors (TreeRing, StableSignature) can pick up M1's pipeline path
    without colliding with the mi-finetuned ``model.*`` written by the
    pipeline chain.
    """
    if not _needs_model_base(config):
        return config

    m1_type = config.get("m1_model_type")
    mi_type = config.get("mi_model_type") or m1_type
    primary_type = mi_type or m1_type
    if not primary_type:
        return config

    primary_cfg = _load_yaml_if_exists(_MODELS_DIR / f"{primary_type}.yaml")
    if primary_cfg is None:
        return config

    layers = []
    if m1_type and m1_type != mi_type:
        m1_cfg = _load_yaml_if_exists(_MODELS_DIR / f"{m1_type}.yaml")
        if m1_cfg is not None:
            m1_promoted = OmegaConf.create({})
            m1_model_block = m1_cfg.get("model") if hasattr(m1_cfg, "get") else None
            if m1_model_block is not None:
                m1_promoted["m1_model"] = m1_model_block
            layers.extend([m1_cfg, m1_promoted])
    layers.append(primary_cfg)
    layers.append(config)
    return OmegaConf.merge(*layers)


def add_common_args(parser) -> None:
    parser.add_argument("base_dir", type=str, default=os.environ.get("RADIOACTIVITY_BASE"))
    parser.add_argument("model_dir", type=str, default=os.path.join(_require_env("RADIOACTIVITY_MODELS"), "models"))
    parser.add_argument("cache_dir", type=str, default=os.path.join(_require_env("RADIOACTIVITY_OUT"), "cache"))


def load_config(
    config_path: str,
    watermark_config_path: str | None = None,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Load a stage config YAML, optionally merged with a watermark config.

    Merge order: ``watermark_config`` → ``config_path`` → ``--model_type`` override
    → configs/models/<type>.yaml.
    The watermark config owns ``watermark.method / type / params`` and any
    watermark-owned resource paths; the stage config supplies everything else
    (plus stage-specific watermark keys like ``watermark.mode``). Stage config
    wins for any overlap with the watermark file. ``model_type``, if given,
    seeds ``m1_model_type`` / ``mi_model_type`` so compose_generation_config
    can pick the right model file; it loses to any value explicitly set in
    ``config_path`` or ``watermark_config_path``.
    """
    config = OmegaConf.load(config_path)
    if watermark_config_path:
        wm_cfg = OmegaConf.load(watermark_config_path)
        config = OmegaConf.merge(wm_cfg, config)
    if model_type and not (config.get("mi_model_type") or config.get("m1_model_type")):
        OmegaConf.set_struct(config, False)
        config["m1_model_type"] = model_type
        config["mi_model_type"] = model_type
    return compose_generation_config(config)


def add_watermark_config_arg(parser) -> None:
    """Register the shared ``--watermark_config`` flag on a stage argparser."""
    parser.add_argument(
        "--watermark_config",
        default=None,
        help="Path to a watermark config (configs/watermarks/<method>.yaml). "
        "Merged under the stage config so the stage can stay generic.",
    )


def add_model_type_arg(parser) -> None:
    """Register the shared ``--model_type`` flag on a stage argparser.

    Supplies ``m1_model_type`` / ``mi_model_type`` for single-stage runs where
    the pipeline isn't injecting them. Ignored if the loaded config already
    supplies a model type (e.g. watermark config, concrete run config).
    """
    parser.add_argument(
        "--model_type",
        default=None,
        help="Model type (e.g. sd21, sd14, sd3, infinity_2b). Selects "
        "configs/models/<type>.yaml to merge in. Only needed when neither "
        "the stage nor watermark config declares m1_model_type / mi_model_type.",
    )

def set_seeds(seed):
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
