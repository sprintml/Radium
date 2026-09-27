"""Central finetuning orchestration for supported model families."""

from __future__ import annotations

import argparse
import logging
import random
import re
from pathlib import Path

from omegaconf import OmegaConf

from pipeline.stage import save_stage_config_snapshot
from src.paths import resolve_auto_paths, build_run_id
from src.finetuning.runner import FinetuningRunner, load_finetune_config

logger = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def get_model_type(config) -> str:
    model_type = config.get("mi_model_type") if hasattr(config, "get") else None
    if not model_type:
        raise ValueError("Missing required key: mi_model_type")
    return str(model_type).lower()


def resolve_finetuning_paths(config) -> dict[str, str | None]:
    output_cfg = config.get("output", {}) or {}
    output_dir = output_cfg.get("output_dir") or None
    if not output_dir:
        try:
            auto = resolve_auto_paths(config)
            output_dir = auto.get("output_dir")
        except ValueError:
            pass
    return {"output_dir": output_dir}


def is_finetuning_complete(config) -> bool:
    output_paths = resolve_finetuning_paths(config)
    out_dir = output_paths.get("output_dir")
    if not out_dir:
        return False
    runners, aliases = _build_registry()
    try:
        model_type = get_model_type(config)
    except ValueError:
        return False
    canonical = aliases.get(model_type.lower())
    if canonical is None:
        return False
    return runners[canonical].is_complete(out_dir)


def _list_image_files(directory: str | None) -> list[Path]:
    if not directory:
        return []
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        return []
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _extract_image_id(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem) or re.search(r"img_(\d+)", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _take_deterministic(files: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    if not files:
        return []
    if count <= len(files):
        return files[:count]
    out: list[Path] = []
    idx = 0
    while len(out) < count:
        out.append(files[idx % len(files)])
        idx += 1
    return out


def _format_image_selection(selected_files: list[Path]) -> str:
    count = len(selected_files)
    if count == 0:
        return "0 images"
    parsed_ids = [_extract_image_id(path) for path in selected_files]
    ids = [idx for idx in parsed_ids if idx is not None]
    if len(ids) != count:
        return f"{count} images"
    return f"images {min(ids)}-{max(ids)} ({count} total)"


def _log_selection(label: str, directory: str | None, selected_files: list[Path], *, shuffle: bool = False) -> None:
    if not directory:
        return
    path = Path(directory)
    if not path.exists():
        logger.warning("%s: directory not found: %s", label, directory)
        return
    shuffle_note = " after deterministic shuffle" if shuffle else ""
    logger.info("%s: %s from %s%s", label, _format_image_selection(selected_files), directory, shuffle_note)


def _log_finetuning_data_selection(config, auto_paths: dict[str, str | None]) -> None:
    """Log which clean/watermarked images will be used.

    Two training sources — clean and watermarked — are mixed according to
    training.watermark_fraction. Both sources are always shuffled
    deterministically from the config seed before taking the top-N slice.
    """
    datasets_cfg = config.get("datasets", {}) or {}
    training_setup = config.get("training_setup", {}) or {}
    training = config.get("training", {}) or {}

    clean_source = str(datasets_cfg.get("finetune_clean_source", "generated")).lower()
    if clean_source in ("coco", "coco_val"):
        clean_dir = datasets_cfg.get("coco_val_image_dir")
    else:
        clean_dir = datasets_cfg.get("train_data_dir_clean") or auto_paths.get("train_data_dir_clean")
    watermarked_dir = datasets_cfg.get("train_data_dir_watermarked") or auto_paths.get("train_data_dir_watermarked")
    val_dir = datasets_cfg.get("validation_data_dir") or auto_paths.get("validation_data_dir")

    train_limit = training_setup.get("max_train_samples")
    val_limit = training_setup.get("max_validation_samples")
    seed = int(config.get("seed", 42) or 42)

    if clean_dir and watermarked_dir:
        # Dual-source mix driven by watermark_fraction.
        total_images = int(train_limit if train_limit is not None else 5000)
        watermark_fraction = float(training.get("watermark_fraction", 1.0))

        if clean_source == "generated":
            # Mirror the runner: disjoint image-id partition.
            from src.finetuning.id_partition import partition_by_image_id
            try:
                clean_selected, watermarked_selected = partition_by_image_id(
                    clean_dir, watermarked_dir,
                    total=total_images,
                    watermark_fraction=watermark_fraction,
                    seed=seed,
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Could not preview disjoint partition: %s", exc)
                clean_selected, watermarked_selected = [], []
            _log_selection(
                "Training clean images (disjoint by image id)",
                clean_dir,
                clean_selected,
                shuffle=True,
            )
            _log_selection(
                "Training watermarked images (disjoint by image id)",
                watermarked_dir,
                watermarked_selected,
                shuffle=True,
            )
        else:
            # coco branch: independent sampling, different ID spaces.
            clean_fraction = 1.0 - watermark_fraction
            clean_files = _list_image_files(clean_dir)
            watermarked_files = _list_image_files(watermarked_dir)
            rng = random.Random(seed)
            rng.shuffle(clean_files)
            rng.shuffle(watermarked_files)

            clean_count = int(round(clean_fraction * total_images))
            watermarked_count = int(round(watermark_fraction * total_images))
            if clean_count + watermarked_count != total_images:
                clean_count = max(0, clean_count + (total_images - clean_count - watermarked_count))

            _log_selection(
                "Training clean images (coco)",
                clean_dir,
                _take_deterministic(clean_files, clean_count),
                shuffle=True,
            )
            _log_selection(
                "Training watermarked images",
                watermarked_dir,
                _take_deterministic(watermarked_files, watermarked_count),
                shuffle=True,
            )
    else:
        # Single-source mode — whichever directory is set.
        single_dir = watermarked_dir or clean_dir
        label = "Training watermarked images" if single_dir is watermarked_dir else "Training images"
        train_files = _list_image_files(single_dir)
        if train_limit is not None:
            train_files = train_files[: min(len(train_files), int(train_limit))]
        _log_selection(label, single_dir, train_files)

    validation_files = _list_image_files(val_dir)
    if val_limit is not None:
        validation_files = validation_files[: min(len(validation_files), int(val_limit))]
    _log_selection("Validation images", val_dir, validation_files)


def _build_registry() -> tuple[dict[str, FinetuningRunner], dict[str, str]]:
    """Return (runners, aliases) by iterating registered ModelBackends.

    To add a new backend, create ``src/architectures/<name>.py`` — there is no
    code in this file to edit per architecture.
    """
    import src.architectures  # noqa: F401  ensure all backends are imported
    from src.architectures.base import _REGISTRY as _BACKENDS

    runners: dict[str, FinetuningRunner] = {}
    aliases: dict[str, str] = {}
    for alias, backend in _BACKENDS.items():
        canonical = backend.name.lower()
        if canonical not in runners:
            runners[canonical] = backend.build_finetuning_runner()
        aliases[alias] = canonical
    return runners, aliases


def main() -> None:
    from src.utils.commonargs import add_watermark_config_arg, add_model_type_arg
    parser = argparse.ArgumentParser(description="Modular finetuning script for diffusion models")
    parser.add_argument("--config_path", type=str, required=True)
    add_watermark_config_arg(parser)
    add_model_type_arg(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_finetune_config(
        args.config_path,
        watermark_config_path=args.watermark_config,
        model_type=args.model_type,
    )
    logger.info("Loaded merged config from %s:\n%s", args.config_path, OmegaConf.to_yaml(config))

    output_paths = resolve_finetuning_paths(config)
    output_dir = output_paths.get("output_dir")
    saved_config = save_stage_config_snapshot(
        config,
        output_dir,
        filename="finetuning_config.yaml",
        overrides={
            "output.output_dir": output_dir,
            "runtime.output_dir": output_dir,
            "runtime.config_path": str(Path(args.config_path).resolve()),
        },
    )
    if saved_config:
        logger.info("Saved config snapshot to %s", saved_config)

    if is_finetuning_complete(config):
        logger.info("Finetuning already complete - skipping.")
        return

    requested_model_type = get_model_type(config)
    runners, aliases = _build_registry()
    canonical = aliases.get(requested_model_type.lower())
    if canonical is None:
        supported = ", ".join(sorted(aliases))
        raise NotImplementedError(
            f"Finetuning for model type '{requested_model_type}' is not supported. Supported values: {supported}"
        )

    auto_paths = resolve_auto_paths(config)
    logger.info("Model type:       %s (requested: %s)", canonical, requested_model_type)
    logger.info("Output directory: %s", output_dir)

    training = config.get("training", {}) or {}
    missing = [k for k in ("watermark_fraction", "num_train_epochs", "learning_rate", "train_batch_size") if training.get(k) is None]
    if missing:
        raise ValueError(
            f"Finetuning config is missing required training keys {missing}. "
            "Ensure the appropriate base under configs/finetuning/ is merged or override them explicitly."
        )
    eff_bs = int(training["train_batch_size"]) * int(training.get("gradient_accumulation_steps", 1) or 1)
    datasets_cfg = config.get("datasets", {}) or {}
    clean_source = datasets_cfg.get("finetune_clean_source", "generated")
    suffix = training.get("run_id_suffix")
    run_id = build_run_id(
        watermark_fraction=float(training["watermark_fraction"]),
        epochs=int(training["num_train_epochs"]),
        learning_rate=float(training["learning_rate"]),
        effective_batch_size=eff_bs,
        clean_source=clean_source,
        suffix=str(suffix) if suffix else None,
    )
    logger.info("run_id:           %s", run_id)

    _log_finetuning_data_selection(config, auto_paths)

    logger.info("Starting finetuning run")
    runners[canonical].execute(config, output_paths)
    logger.info("Finetuning complete")


class FinetuningStage:
    name = "finetuning"

    def load_config(self, config_path: str, watermark_config_path: str | None = None):
        return load_finetune_config(config_path, watermark_config_path=watermark_config_path)

    def is_complete(self, config, command: list[str]) -> bool:
        return is_finetuning_complete(config)


STAGE = FinetuningStage()


if __name__ == "__main__":
    main()
