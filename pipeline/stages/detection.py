"""Watermark detection: scores a single image directory and caches raw results.

Each detection step scores *every* image in one image set (M1 clean or Mi
generated) for a specific prompt set, writing one entry per image to
``<output_dir>/scores/<wm_name>.json``.

The ``detection.target`` config key selects the image source:
  - ``m1_clean``: M1 clean-generated images (no watermark).
  - ``mi``:       Mi finetuned-model generated images.

The ``watermark.iteration`` key selects the prompt set index, and
``datasets.set_name`` overrides the resolved prompt-set name when set.
The Mi model iteration is always ``watermark.iteration + 1``.

Score caches are consumed by pipeline/stages/evaluation.py. Per-seed
random sub-sampling lives there, not here — detection is a single
deterministic pass over all images in the source directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

from pipeline.stage import save_stage_config_snapshot
from pipeline.eval_common import (
    evaluate_images,
    get_watermark_from_config,
    load_score_map,
    save_score_map,
    score_cache_path,
)
from pipeline.feature_cache import (
    feature_cache_indices,
    feature_cache_path,
    save_feature_map,
)
from src.utils import load_config


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_detection_paths(config) -> dict[str, str | None]:
    """Resolve image_dir and output_dir from detection_target + config.

    Returns {"image_dir": ..., "output_dir": ...} with None values when
    insufficient config keys are present.
    """
    from src.paths import (
        ProjectPaths,
        prompt_set_name,
        resolve_m1_model_type,
        resolve_mi_model_type,
        resolve_wm_params,
    )

    wm = (config.get("watermark", {}) or {}) if hasattr(config, "get") else {}
    ds = (config.get("datasets", {}) or {}) if hasattr(config, "get") else {}
    detection_cfg = (config.get("detection", {}) or {}) if hasattr(config, "get") else {}
    target = detection_cfg.get("target") or "m1_clean"

    method = wm.get("method")
    params = resolve_wm_params(config)
    m1_model_type = resolve_m1_model_type(config)
    mi_model_type = resolve_mi_model_type(config)
    run_id = wm.get("run_id")
    iteration = int(wm.get("iteration", 1))
    model_iteration = iteration + 1
    dataset = ds.get("name") or "coco"
    set_name = ds.get("set_name") or prompt_set_name(iteration)

    if not m1_model_type:
        return {"image_dir": None, "output_dir": None}

    try:
        layout = ProjectPaths.from_config(config)
    except ValueError:
        return {"image_dir": None, "output_dir": None}

    if target == "m1_clean":
        image_dir = str(layout.clean_generated(m1_model_type, dataset, set_name))
        output_dir = str(layout.clean_evaluations(m1_model_type, dataset, set_name))
        return {"image_dir": image_dir, "output_dir": output_dir}

    if target == "m1_watermarked":
        if not all([method, params]):
            return {"image_dir": None, "output_dir": None}
        image_dir = str(layout.m1_generated(m1_model_type, method, params, dataset, set_name))
        output_dir = str(layout.m1_evaluations(m1_model_type, method, params, dataset, set_name))
        return {"image_dir": image_dir, "output_dir": output_dir}

    if target == "mi":
        if not all([method, params, run_id, mi_model_type]):
            return {"image_dir": None, "output_dir": None}
        image_dir = str(
            layout.mi_generated(m1_model_type, method, params, run_id, model_iteration, mi_model_type, dataset, set_name)
        )
        output_dir = str(
            layout.mi_evaluations(m1_model_type, method, params, run_id, model_iteration, mi_model_type, dataset, set_name)
        )
        return {"image_dir": image_dir, "output_dir": output_dir}

    raise ValueError(
        f"Unknown detection.target '{target}'. Must be 'm1_clean', 'm1_watermarked', or 'mi'."
    )


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _list_indexed_images(image_dir: str) -> list[tuple[int, str]]:
    """Return [(index, filename), ...] sorted by index for all images in `image_dir`.

    Raises ValueError on any filename that doesn't match the expected
    ``<prefix>_<index>.<ext>`` pattern — detection keys the cache by index,
    so an unparsable filename is a hard error rather than a silent skip.
    """
    items: list[tuple[int, str]] = []
    for f in os.listdir(image_dir):
        if Path(f).suffix.lower() not in _IMAGE_EXTS:
            continue
        try:
            idx = int(f.split("_")[1].split(".")[0])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Cannot parse image index from {f!r} in {image_dir}; "
                "expected a filename of the form '<prefix>_<index>.<ext>'."
            ) from exc
        items.append((idx, f))
    items.sort(key=lambda t: t[0])
    return items


def is_detection_complete(config, output_dir: str) -> bool:
    """Return True if the score cache exists and covers every image in the source dir.

    When ``detection.save_features`` is true, the feature cache must also cover
    every image — otherwise a partial rerun won't refill missing features.
    """
    if not output_dir:
        return False
    wm = (config.get("watermark", {}) or {}) if hasattr(config, "get") else {}
    detection_cfg = (config.get("detection", {}) or {}) if hasattr(config, "get") else {}
    wm_name = wm.get("method") or "unknown"

    cache_path = score_cache_path(os.path.join(output_dir, "scores"), wm_name)
    if not cache_path.exists():
        return False

    image_dir = resolve_detection_paths(config).get("image_dir")
    if not image_dir or not os.path.isdir(image_dir):
        return False

    try:
        score_map = load_score_map(cache_path)
        needed = {idx for idx, _ in _list_indexed_images(image_dir)}
    except (ValueError, OSError, json.JSONDecodeError):
        return False
    if not (bool(needed) and needed.issubset(score_map.keys())):
        return False

    if bool(detection_cfg.get("save_features", False)):
        feature_mode = wm.get("feature_cache_mode")
        if not feature_mode:
            raise ValueError(
                "detection.save_features=true requires watermark.feature_cache_mode "
                "to be set in the watermark config."
            )
        feat_path = feature_cache_path(
            os.path.join(output_dir, "features"), wm_name, mode=feature_mode
        )
        try:
            covered = feature_cache_indices(feat_path, mode=feature_mode)
        except (OSError, ValueError, NotImplementedError):
            return False
        if not needed.issubset(covered):
            return False
    return True


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def detect_single_dir(config, output_dir: str, force: bool = False) -> None:
    """Score every image in the source directory once and cache by index.

    When ``force`` is True the existing per-watermark score cache is deleted
    first so every image is re-scored, even if the cache is already complete.
    """
    config_base = dict(config)
    wm_cfg = config_base.get("watermark", {}) or {}
    datasets_cfg = config_base.get("datasets", {}) or {}
    dataset_params = config_base.get("dataset_params", {}) or {}
    detection_cfg = dict(config_base.get("detection", {}) or {})
    batch_size = int(detection_cfg.get("batch_size", 32))
    save_features = bool(detection_cfg.get("save_features", False))
    feature_mode: str | None = wm_cfg.get("feature_cache_mode")
    if save_features and not feature_mode:
        raise ValueError(
            "detection.save_features=true requires watermark.feature_cache_mode "
            "to be set in the watermark config."
        )

    paths = resolve_detection_paths(config)
    image_dir = paths.get("image_dir")
    if not image_dir:
        raise ValueError(
            "Could not resolve image_dir. Check detection.target and watermark config."
        )

    wm_name = wm_cfg.get("method") or "unknown"
    scores_dir = os.path.join(output_dir, "scores")
    cache_path = score_cache_path(scores_dir, wm_name)

    features_dir = os.path.join(output_dir, "features") if save_features else None
    feat_path = (
        feature_cache_path(features_dir, wm_name, mode=feature_mode)
        if save_features else None
    )

    if force and cache_path.exists():
        cache_path.unlink()
        logger.info("--force: deleted existing score cache %s", cache_path)
    if force and save_features and feat_path.exists():
        if feat_path.is_dir():
            import shutil
            shutil.rmtree(feat_path)
        else:
            feat_path.unlink()
        logger.info("--force: deleted existing feature cache %s", feat_path)

    existing: dict = {}
    if cache_path.exists():
        existing = load_score_map(cache_path)

    existing_features: set[int] = set()
    if save_features:
        existing_features = feature_cache_indices(feat_path, mode=feature_mode)

    items = _list_indexed_images(image_dir)
    if save_features:
        todo = [
            (idx, f) for idx, f in items
            if idx not in existing or idx not in existing_features
        ]
    else:
        todo = [(idx, f) for idx, f in items if idx not in existing]
    if not todo:
        logger.info(
            "Detection cache already covers all %d images in %s", len(items), image_dir
        )
        return

    # Reuse the generation model config for detector loading: watermark
    # classes that need a model (e.g. TreeRing's inversion pipeline) read
    # model.model_path from here. compose_generation_config has already
    # merged the base-model YAML when the config was loaded.
    model_cfg = config_base.get("model", {}) or {}
    full_config = OmegaConf.create({
        "watermark": wm_cfg,
        "datasets": {**datasets_cfg, "output_dir": image_dir},
        "dataset_params": dataset_params,
        "model": model_cfg,
        "m1_model_type": config_base.get("m1_model_type"),
        "mi_model_type": config_base.get("mi_model_type"),
    })
    watermark, _ = get_watermark_from_config(full_config)

    from PIL import Image as PILImage

    logger.info(
        "Detecting %d images (of %d total) from %s → %s",
        len(todo), len(items), image_dir, cache_path,
    )
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        imgs = [
            PILImage.open(os.path.join(image_dir, f)).convert("RGB")
            for _, f in chunk
        ]
        if save_features:
            scores, feats = evaluate_images(watermark, imgs, save_features=True)
            if len(feats) != len(chunk):
                raise ValueError(
                    f"detect_features returned {len(feats)} vectors for a batch of "
                    f"{len(chunk)} images on {type(watermark).__name__}."
                )
            save_feature_map(
                feat_path,
                {idx: feats[k] for k, (idx, _) in enumerate(chunk)},
                mode=feature_mode,
            )
            # Score cache is strictly additive when topping up features: only
            # write entries for indices not already covered, so re-running with
            # save_features=True on an existing scored scenario leaves the
            # original score cache contents (and mtime) untouched.
            entries = {
                idx: {"score": float(s)}
                for (idx, _), s in zip(chunk, scores)
                if idx not in existing
            }
            if entries:
                save_score_map(cache_path, entries)
        else:
            scores = evaluate_images(watermark, imgs)
            entries = {idx: {"score": float(s)} for (idx, _), s in zip(chunk, scores)}
            save_score_map(cache_path, entries)

        logger.info(
            "  Scored %d / %d", min(start + len(chunk), len(todo)), len(todo)
        )


# ---------------------------------------------------------------------------
# Stage descriptor
# ---------------------------------------------------------------------------

class DetectionStage:
    name = "detection"

    def load_config(self, config_path: str, watermark_config_path: str | None = None):
        return load_config(config_path, watermark_config_path=watermark_config_path)

    def is_complete(self, config, command: list[str]) -> bool:
        from pipeline.stage import extract_arg
        output_dir = extract_arg(command, "--output_dir") or resolve_detection_paths(config).get("output_dir")
        if not output_dir:
            return False
        return is_detection_complete(config, output_dir)


STAGE = DetectionStage()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from src.utils.commonargs import add_watermark_config_arg, add_model_type_arg
    parser = argparse.ArgumentParser(description="Watermark detection — score one image directory")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the existing per-watermark score cache and re-score every "
        "image, ignoring is_detection_complete.",
    )
    add_watermark_config_arg(parser)
    add_model_type_arg(parser)
    args = parser.parse_args()

    config = load_config(
        args.config_path,
        watermark_config_path=args.watermark_config,
        model_type=args.model_type,
    )
    logger.info("Loaded config from %s:\n%s", args.config_path, OmegaConf.to_yaml(config))

    output_dir = args.output_dir or resolve_detection_paths(config).get("output_dir")
    if not output_dir:
        raise ValueError(
            "output_dir could not be resolved. "
            "Provide --output_dir or set watermark.run_id (for mi) in config."
        )
    logger.info("Output directory: %s", output_dir)

    saved_config = save_stage_config_snapshot(
        config,
        output_dir,
        filename="detection_config.yaml",
        overrides={
            "runtime.output_dir": output_dir,
            "runtime.config_path": str(Path(args.config_path).resolve()),
        },
    )
    if saved_config:
        logger.info("Saved config snapshot to %s", saved_config)

    force = bool(getattr(args, "force", False))
    if not force and is_detection_complete(config, output_dir):
        logger.info("Detection already complete — skipping.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        detect_single_dir(config, output_dir, force=force)
        logger.info("Detection complete.")
