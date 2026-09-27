"""Shared helpers for comprehensive evaluation scripts."""

from __future__ import annotations

import glob
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

import numpy as np

from pipeline.score_cache import (
    _is_legacy_cache,
    load_score_map,
    save_score_map,
    score_cache_path,
    summarize_score_cache,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def get_watermark_from_config(config):
    from src.watermarks import build_watermark
    return build_watermark(config)


def evaluate_images(watermark, images: List, save_features: bool = False):
    """Score `images` and optionally also return per-image feature vectors.

    When ``save_features`` is False (default), returns ``list[float]`` exactly
    as before. When True, returns ``(scores, features)`` where features is a
    list of per-image numpy arrays — derived from a single inference pass via
    ``detect_features`` + ``scores_from_features`` on watermarks that support
    aggregation. Watermarks without those overrides raise NotImplementedError.
    """
    if not images:
        raise ValueError("images list cannot be empty")

    if save_features:
        try:
            features = watermark.detect_features(images)
            scores = [float(s) for s in watermark.scores_from_features(features)]
        except Exception as e:
            raise RuntimeError(
                f"Error evaluating images with features on {type(watermark).__name__}: {e}"
            ) from e
        return scores, features

    if watermark.supports_batch_evaluate:
        try:
            return list(watermark.detect(images))
        except Exception as e:
            logger.debug("Failed image batch for %s: %s", type(watermark).__name__, images)
            raise RuntimeError(f"Error evaluating images with {type(watermark).__name__}: {e}") from e

    scores = []
    for img in images:
        score = watermark.detect(img)
        scores.append(float(score))
    return scores


def find_watermarked_dir(template: str, percentage: int, mode: str, override: str | None = None) -> str:
    pattern = override.format(mode=mode) if override else template.format(percentage=percentage, mode=mode)
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No directories found matching pattern: {pattern}")
    return matches[0]


def resolve_clean_generated_dir(config_base: dict, set_name: str) -> str:
    """Resolve M1 clean-generated directory from ProjectPaths for the given prompt set.

    Uses the resolved M1 model label and paths.out_base / RADIOACTIVITY_OUT.
    Raises ValueError if required config keys are missing.
    """
    from src.paths import ProjectPaths, resolve_m1_model_type

    m1_model_type = resolve_m1_model_type(config_base)
    if not m1_model_type:
        raise ValueError("Could not resolve the M1 model type for the clean generated directory")

    datasets_cfg = (config_base.get("datasets", {}) or {}) if hasattr(config_base, "get") else {}
    dataset = datasets_cfg.get("name")
    if not dataset:
        raise ValueError("datasets.name is required to resolve the clean generated directory")

    layout = ProjectPaths.from_config(config_base)
    return str(layout.clean_generated(m1_model_type, dataset, set_name))


def resolve_mi_generated_dir(
    config_base: dict,
    finetuning_config,
    dataset: str,
    set_name: str,
) -> str | None:
    """Derive Mi generated-images directory from ProjectPaths when possible.

    Uses ``finetuning_config`` to read watermark/training parameters and
    ``config_base`` for the base path.  Returns None when not enough info
    is available (callers should fall back to explicit paths).
    """
    if finetuning_config is None:
        return None

    try:
        from src.paths import (
            ProjectPaths,
            build_run_id,
            resolve_m1_model_type,
            resolve_wm_params,
        )
    except ImportError:
        return None

    try:
        layout = ProjectPaths.from_config(finetuning_config)
    except ValueError:
        try:
            layout = ProjectPaths.from_config(config_base)
        except ValueError:
            return None

    wm_cfg = (finetuning_config.get("watermark", {}) or {}) if hasattr(finetuning_config, "get") else {}
    method = wm_cfg.get("method")
    wm_params = resolve_wm_params(finetuning_config)
    iteration = int(wm_cfg.get("iteration", 2))
    model_type = resolve_m1_model_type(finetuning_config)

    training = (finetuning_config.get("training", {}) or {}) if hasattr(finetuning_config, "get") else {}
    missing = [k for k in ("watermark_fraction", "num_train_epochs", "learning_rate", "train_batch_size") if training.get(k) is None]
    if missing:
        raise ValueError(
            f"_derive_mi_generated_dir: finetuning_config.training is missing {missing}. "
            "Pass the merged finetuning config (stablediffusion/common.yaml/infinity base + run config)."
        )
    watermark_fraction = float(training["watermark_fraction"])
    epochs = int(training["num_train_epochs"])
    lr = float(training["learning_rate"])
    bs = int(training["train_batch_size"])
    ga = int(training.get("gradient_accumulation_steps", 1) or 1)
    eff_bs = bs * ga

    if not all([method, wm_params, model_type]):
        return None

    datasets_cfg = (finetuning_config.get("datasets", {}) or {}) if hasattr(finetuning_config, "get") else {}
    clean_source = datasets_cfg.get("finetune_clean_source", "generated")
    run_id = build_run_id(
        watermark_fraction=watermark_fraction,
        epochs=epochs,
        learning_rate=lr,
        effective_batch_size=eff_bs,
        clean_source=clean_source,
    )
    return str(
        layout.mi_generated(model_type, method, wm_params, run_id, iteration, model_type, dataset, set_name)
    )


def infer_paths_from_config(config, wm_cfg, force_clean=None, force_wm=None) -> Tuple[str | None, str | None]:
    """Infer clean and watermarked directories from config."""
    datasets = config.get("datasets", {})
    clean_dir_cfg = datasets.get("clean_dir")
    watermarked_dir_cfg = datasets.get("watermarked_dir")
    input_dir = datasets.get("input_dir")
    output_dir = datasets.get("output_dir")

    wm_type = wm_cfg.get("type")
    clean_dir = force_clean if force_clean is not None else clean_dir_cfg
    watermarked_dir = force_wm if force_wm is not None else watermarked_dir_cfg

    if wm_type == "post_generation":
        if clean_dir is None:
            clean_dir = input_dir
        if watermarked_dir is None:
            watermarked_dir = output_dir
    elif wm_type == "in_generation":
        mode = wm_cfg.get("mode", None)
        if mode == "clean_generate":
            if clean_dir is None:
                clean_dir = output_dir
        elif mode == "generate":
            if watermarked_dir is None:
                watermarked_dir = output_dir

    return clean_dir, watermarked_dir


def fnames_to_indices(fnames: List[str]) -> List[int]:
    """Parse global dataset indices from filenames like img_005042.png."""
    indices = []
    for fname in fnames:
        try:
            indices.append(int(fname.split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            pass
    return indices


# ---------------------------------------------------------------------------
# Score cache helpers (re-exported from pipeline.score_cache so this module
# stays the single shared dependency for stage code, while the planner can
# import score_cache directly without pulling watermark/diffusion backends).
# ---------------------------------------------------------------------------


def sample_scores(
    score_map: Dict[int, Dict[str, Any]], seed: int, n: int
) -> List[float]:
    """Independent random sample of up to `n` scores, seeded by `seed`."""
    indices = sorted(score_map.keys())
    k = min(n, len(indices))
    picked = random.Random(seed).sample(indices, k)
    return [float(score_map[i]["score"]) for i in picked]


def sample_paired_scores(
    w_map: Dict[int, Dict[str, Any]],
    clean_map: Dict[int, Dict[str, Any]],
    seed: int,
    n: int,
) -> Tuple[List[float], List[float]]:
    """Paired-by-index sample from Mi and M1 caches (supervised eval).

    Raises ValueError if the two index sets differ — the caller is expected
    to regenerate/re-detect the offending side rather than silently proceed
    with the intersection.
    """
    w_ids = set(w_map.keys())
    c_ids = set(clean_map.keys())
    if w_ids != c_ids:
        missing_from_clean = sorted(w_ids - c_ids)[:5]
        missing_from_w = sorted(c_ids - w_ids)[:5]
        raise ValueError(
            "Supervised evaluation requires identical index sets in the Mi "
            f"and M1 score caches, got |Mi|={len(w_ids)} and |M1|={len(c_ids)}. "
            f"First few missing from M1: {missing_from_clean}. "
            f"First few missing from Mi: {missing_from_w}."
        )
    indices = sorted(w_ids)
    k = min(n, len(indices))
    picked = random.Random(seed).sample(indices, k)
    w_scores = [float(w_map[i]["score"]) for i in picked]
    c_scores = [float(clean_map[i]["score"]) for i in picked]
    return w_scores, c_scores


