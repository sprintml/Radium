"""Finetuning runner for Infinity via torchrun subprocess."""

from __future__ import annotations

import json
import logging
import os
import random
import socket
import subprocess
import sys
from pathlib import Path

from src.finetuning.runner import FinetuningRunner, assert_input_dir
from src.finetuning.id_partition import partition_by_image_id
from src.paths import resolve_auto_paths
from src.utils.commonargs import cfg_get as _cfg_get

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_DEFAULT_H_DIV_W = "1.000"


def _flag_value(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def _find_free_port() -> int:
    # Avoids torch.distributed's default 29500 when multiple finetune jobs
    # land on the same node — small race window vs. guaranteed collision.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _read_prompts(image_dir: Path) -> dict[str, str]:
    metadata = image_dir / "metadata.jsonl"
    prompts: dict[str, str] = {}
    if not metadata.exists():
        logger.warning("No metadata.jsonl in %s — entries will have empty prompts", image_dir)
        return prompts
    for line in metadata.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        fname = entry.get("file_name") or ""
        if fname:
            prompts[fname] = entry.get("text", "")
    return prompts


def _select_entries(image_dir: str | None, count: int, *, seed: int, h_div_w: float = 1.0) -> list[dict]:
    if not image_dir or count <= 0:
        return []
    path = Path(image_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Infinity training source not found: {image_dir}")
    files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    selected: list[Path] = []
    idx = 0
    while len(selected) < count:
        selected.append(files[idx % len(files)])
        idx += 1
    prompts = _read_prompts(path)
    return _entries_from_files(selected, prompts, h_div_w=h_div_w)


def _select_entries_with_prompts(
    image_dir: str,
    count: int,
    *,
    seed: int,
    prompts: dict[str, str],
    h_div_w: float = 1.0,
) -> list[dict]:
    """Like _select_entries, but caption lookup uses an explicit prompts dict."""
    if count <= 0:
        return []
    path = Path(image_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Infinity training source not found: {image_dir}")
    files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    selected: list[Path] = []
    idx = 0
    while len(selected) < count:
        selected.append(files[idx % len(files)])
        idx += 1
    return _entries_from_files(selected, prompts, h_div_w=h_div_w)


def _entries_from_files(
    files: list[Path],
    prompts: dict[str, str],
    *,
    h_div_w: float = 1.0,
) -> list[dict]:
    entries: list[dict] = []
    for p in files:
        prompt = prompts.get(p.name, "") or prompts.get(str(p.resolve()), "") or prompts.get(p.stem, "")
        entries.append({
            "image_path": str(p.resolve()),
            "h_div_w": h_div_w,
            "long_caption": prompt,
            "long_caption_type": "caption-InternVL2.0",
            "text": prompt,
            "short_caption_type": "short caption-InternVL2.0",
        })
    return entries


class InfinityRunner(FinetuningRunner):
    def is_complete(self, output_dir: str) -> bool:
        model_dir = Path(output_dir) / "model"
        return model_dir.exists() and (
            any(model_dir.glob("*.pt")) or any(model_dir.glob("*.pth"))
        )

    def _build_manifest(self, config, output_paths: dict[str, str | None]) -> str:
        """Build the Infinity-style jsonl manifest folder and return its path.

        Mixes clean and watermarked sources by training.watermark_fraction
        deterministically from the config seed, mirroring the SD-family
        selection logic in pipeline/stages/finetuning.py.
        """
        out_dir = output_paths.get("output_dir")
        if not out_dir:
            raise ValueError("Infinity finetuning requires output_paths['output_dir']")
        manifest_dir = Path(out_dir) / "finetune_manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        for stale in manifest_dir.glob("*.jsonl"):
            stale.unlink()

        training = config.get("training", {}) or {}
        training_setup = config.get("training_setup", {}) or {}
        datasets_cfg = config.get("datasets", {}) or {}
        seed = int(_cfg_get(config, "seed", 42) or 42)

        auto_paths = resolve_auto_paths(config)
        watermarked_dir = (
            datasets_cfg.get("train_data_dir_watermarked")
            or auto_paths.get("train_data_dir_watermarked")
        )

        train_limit = training_setup.get("max_train_samples")
        total_images = int(train_limit) if train_limit is not None else 5000
        watermark_fraction = float(training.get("watermark_fraction", 1.0))
        clean_source = str(datasets_cfg.get("finetune_clean_source", "generated")).lower()

        if clean_source == "coco":
            clean_dir = datasets_cfg.get("coco_val_image_dir")
            if not clean_dir:
                raise ValueError(
                    "datasets.coco_val_image_dir must be set when "
                    "datasets.finetune_clean_source=coco"
                )
            annotation_path = datasets_cfg.get("coco_val_annotation")
            if not annotation_path:
                raise ValueError(
                    "datasets.coco_val_annotation must be set when "
                    "datasets.finetune_clean_source=coco"
                )
        else:
            clean_dir = (
                datasets_cfg.get("train_data_dir_clean")
                or auto_paths.get("train_data_dir_clean")
            )
            annotation_path = None

        if clean_dir and watermarked_dir:
            # Pre-flight: fail fast if upstream stages didn't write the
            # directories we expect, so the failure points at the actual
            # cause instead of crashing later inside the partitioner.
            assert_input_dir("training watermarked images", watermarked_dir)
            assert_input_dir("training clean images", clean_dir)

            watermarked_count = int(round(watermark_fraction * total_images))
            clean_count = total_images - watermarked_count

            if clean_source == "coco":
                # Different ID spaces (real COCO image_ids vs img_{idx}.png) — no
                # dedup needed, just sample independently.
                from src.datasets.loader import load_coco_val_as_metadata
                clean_prompts = load_coco_val_as_metadata(
                    annotation_path, clean_dir, seed=seed
                )
                entries = _select_entries_with_prompts(
                    clean_dir, clean_count, seed=seed, prompts=clean_prompts
                )
                entries += _select_entries(watermarked_dir, watermarked_count, seed=seed)
            else:
                clean_files, watermarked_files = partition_by_image_id(
                    clean_dir, watermarked_dir,
                    total=total_images,
                    watermark_fraction=watermark_fraction,
                    seed=seed,
                )
                entries = _entries_from_files(clean_files, _read_prompts(Path(clean_dir)))
                entries += _entries_from_files(
                    watermarked_files, _read_prompts(Path(watermarked_dir))
                )
        else:
            single_dir = watermarked_dir or clean_dir
            if not single_dir:
                raise ValueError(
                    "Infinity finetuning needs datasets.train_data_dir_clean or "
                    "datasets.train_data_dir_watermarked (or resolvable auto paths)."
                )
            entries = _select_entries(single_dir, total_images, seed=seed)

        random.Random(seed).shuffle(entries)

        out_path = manifest_dir / f"{_DEFAULT_H_DIV_W}_{len(entries)}.jsonl"
        with out_path.open("w") as fp:
            for entry in entries:
                fp.write(json.dumps(entry) + "\n")
        logger.info(
            "Wrote Infinity manifest %s (%d samples; watermark_fraction=%.3f)",
            out_path, len(entries), watermark_fraction,
        )
        return str(manifest_dir)

    def _run(self, config, output_paths: dict[str, str | None]) -> None:
        strategy = str(_cfg_get(config, "finetuning.launch.strategy", "auto")).lower()
        if strategy not in ("auto", "torchrun"):
            raise ValueError(
                f"Infinity backend requires finetuning.launch.strategy in ['auto', 'torchrun'], got: {strategy}"
            )
        manifest_dir = self._build_manifest(config, output_paths)
        command = self._build_command(config, output_paths, manifest_dir)
        logger.info("Running Infinity finetuning command: %s", " ".join(command))
        subprocess.run(command, check=True)

        # Select best vs last checkpoint for downstream stages, mirroring the
        # SD runner's finetuning.use_best_model flag.
        out_dir = output_paths.get("output_dir")
        if out_dir:
            model_dir = Path(out_dir) / "model"
            use_best = bool(_cfg_get(config, "finetuning.use_best_model", False))
            src_name = "ar-ckpt-best.pth" if use_best else "ar-ckpt-last.pth"
            src_path = model_dir / src_name
            link_path = model_dir / "ar-ckpt-selected.pth"
            if src_path.exists():
                if link_path.is_symlink() or link_path.exists():
                    link_path.unlink()
                os.symlink(src_name, link_path)
            else:
                logger.warning("Infinity: %s not found in %s — leaving selection unset", src_name, model_dir)

    def _build_command(
        self,
        config,
        output_paths: dict[str, str | None],
        manifest_dir: str,
    ) -> list[str]:
        training = config.get("training", {}) or {}
        training_setup = config.get("training_setup", {}) or {}
        nproc = int(_cfg_get(config, "finetuning.launch.torchrun.nproc_per_node", 1))
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}"]
        master_port = _cfg_get(config, "finetuning.launch.torchrun.master_port", None)
        if master_port is None:
            master_port = _find_free_port()
        cmd.append(f"--master_port={master_port}")
        cmd.append("src/finetuning/infinity/train.py")

        out_dir = output_paths.get("output_dir")
        log_dir = str(Path(out_dir) / "logs") if out_dir else None
        bed_path = str(Path(out_dir) / "model") if out_dir else None
        if bed_path:
            Path(bed_path).mkdir(parents=True, exist_ok=True)

        # rush_resume can be a directory (Mi output, injected by planner) or
        # a file (base release). Resolve via the backend; train.py handles
        # the trainer-vs-flat format itself.
        rush_resume = _cfg_get(config, "model.pretrained_model_name_or_path")
        if rush_resume:
            from src.architectures.base import get_backend
            rush_resume = get_backend("infinity_2b").resolve_checkpoint_path(rush_resume)

        args: dict[str, object] = {
            "ep": training.get("num_train_epochs", _cfg_get(config, "finetuning.mapped_args.ep", 2)),
            "lbs": training.get("train_batch_size", _cfg_get(config, "finetuning.mapped_args.lbs", 2)),
            "cum": training_setup.get(
                "gradient_accumulation_steps",
                _cfg_get(config, "finetuning.mapped_args.cum", 1),
            ),
            "tblr": training.get("learning_rate", _cfg_get(config, "finetuning.mapped_args.tblr", 6e-4)),
            # data_path points to the manifest folder produced by _build_manifest.
            # Infinity's T2IIterableDataset globs *.jsonl in this folder.
            "data_path": manifest_dir,
            "local_out_path": log_dir,
            "bed": bed_path,
            "exp_name": _cfg_get(
                config, "run.id", _cfg_get(config, "finetuning.mapped_args.exp_name", "infinity_finetune")
            ),
            "rush_resume": rush_resume,
        }

        override_args = _cfg_get(config, "backend_overrides.infinity.args", {}) or {}
        args.update(override_args)
        raw_args = _cfg_get(config, "backend_overrides.infinity.raw_args", []) or []

        for k, v in args.items():
            if v is None:
                continue
            cmd.extend([f"--{k}", _flag_value(v)])
        for item in raw_args:
            cmd.append(str(item))
        return cmd
