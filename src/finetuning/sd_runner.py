"""Finetuning runner for Stable Diffusion model families."""

from __future__ import annotations

from typing import Callable

from src.finetuning.runner import FinetuningRunner, _apply_attr_dict, assert_input_dir
from src.utils.commonargs import cfg_get as _cfg_get
class SDRunner(FinetuningRunner):
    """Runs SD-family finetuning in-process via a module's main_with_args."""

    def __init__(self, model_name: str, module_loader: Callable):
        self.model_name = model_name
        self._module_loader = module_loader

    def is_complete(self, output_dir: str) -> bool:
        from pathlib import Path
        return (Path(output_dir) / "model" / "model_index.json").exists()

    def _default_args(self, config) -> dict[str, object]:
        args = dict(_cfg_get(config, "defaults.sd_args_base", {}) or {})
        # finetuning.* supplies tracker_project_name etc. from configs/models/<name>.yaml.
        args.update(dict(_cfg_get(config, "finetuning", {}) or {}))
        return args

    def _run(self, config, output_paths: dict[str, str | None]) -> None:
        from src.paths import resolve_auto_paths

        # Pre-flight: validate input dirs before model load so a missing
        # upstream directory fails fast with a clear message instead of
        # crashing minutes later inside id_partition / dataset loading.
        datasets_cfg = config.get("datasets", {}) or {}
        auto_paths = resolve_auto_paths(config)
        finetune_clean_source = str(datasets_cfg.get("finetune_clean_source", "generated")).lower()
        watermarked_dir = datasets_cfg.get("train_data_dir_watermarked") or auto_paths.get("train_data_dir_watermarked")
        if finetune_clean_source in ("coco", "coco_val"):
            assert_input_dir("training watermarked images", watermarked_dir)
            assert_input_dir("coco_val_image_dir", datasets_cfg.get("coco_val_image_dir"))
        else:
            assert_input_dir("training watermarked images", watermarked_dir)
            clean_dir = datasets_cfg.get("train_data_dir_clean") or auto_paths.get("train_data_dir_clean")
            assert_input_dir("training clean images", clean_dir)

        module = self._module_loader()

        class Args:
            pass

        args = Args()
        _apply_attr_dict(args, self._default_args(config))
        # Auto-derived paths (computed in the pre-flight above) fill in
        # anything not explicitly set in the config.
        _apply_attr_dict(args, auto_paths)

        model_cfg = config.get("model", {}) or {}
        pretrained = model_cfg.get("model_path")
        if not pretrained:
            raise ValueError(
                f"Missing model path for '{self.model_name}'. "
                "Set model.model_path (local path or HF repo id) in configs/models/<name>.yaml."
            )
        args.pretrained_model_name_or_path = pretrained
        args.revision = model_cfg.get("revision") or None
        args.variant = model_cfg.get("variant") or None

        training_setup = config.get("training_setup", {}) or {}
        training = config.get("training", {}) or {}
        # Pass through everything except the semantic training-dir keys — those
        # get translated to the SD script's legacy positional names below.
        _datasets_passthrough = {
            k: v for k, v in dict(datasets_cfg).items()
            if k not in ("train_data_dir_clean", "train_data_dir_watermarked")
        }
        _apply_attr_dict(args, _datasets_passthrough)
        _apply_attr_dict(args, dict(training_setup))

        # Translate the semantic training-data keys to the SD script's expected
        # args.  Two public keys in the config:
        #   datasets.train_data_dir_clean       — unwatermarked source
        #   datasets.train_data_dir_watermarked — watermarked source
        # Dual-folder mixing is used whenever both are set; the mix ratio comes
        # from training.watermark_fraction (1.0 = all watermarked, 0.0 = all
        # clean, 0<f<1 = mix). No separate use_dual_dataset / dual_shuffle flags.
        # clean_dir / watermarked_dir / finetune_clean_source were resolved in
        # the pre-flight at the top of _run.
        clean_dir = datasets_cfg.get("train_data_dir_clean") or auto_paths.get("train_data_dir_clean")

        if finetune_clean_source in ("coco", "coco_val"):
            coco_val_image_dir = datasets_cfg.get("coco_val_image_dir")
            if not coco_val_image_dir:
                raise ValueError(
                    "datasets.coco_val_image_dir must be set when datasets.finetune_clean_source=coco"
                )
            args.train_data_dir = coco_val_image_dir     # folder1 = real COCO val images
            args.train_data_dir_2 = watermarked_dir      # folder2 = watermarked
            args.use_dual_dataset = True
            # Normalize for downstream: SD modules check this string.
            args.finetune_clean_source = "coco_val"
            coco_val_annotation = datasets_cfg.get("coco_val_annotation")
            if coco_val_annotation:
                args.coco_val_annotation = coco_val_annotation
        elif clean_dir and watermarked_dir:
            # generated mode: split img_{idx}.png ids into disjoint clean/watermarked
            # slices sized by training.watermark_fraction so the model never sees
            # both variants of the same prompt.
            wm_fraction = float(training.get("watermark_fraction", 1.0))
            if wm_fraction >= 1.0:
                # All-watermarked: no clean side to mix in. DualFolderDataset rejects
                # empty file lists, so fall through to single-folder mode.
                args.train_data_dir = watermarked_dir
                args.train_data_dir_2 = None
                args.use_dual_dataset = False
            elif wm_fraction <= 0.0:
                # All-clean: no watermarked side to mix in.
                args.train_data_dir = clean_dir
                args.train_data_dir_2 = None
                args.use_dual_dataset = False
            else:
                from src.finetuning.id_partition import partition_by_image_id
                seed_val = int(_cfg_get(config, "seed", 42) or 42)
                total_images = int(training_setup.get("max_train_samples") or 5000)
                clean_files, watermarked_files = partition_by_image_id(
                    clean_dir, watermarked_dir,
                    total=total_images,
                    watermark_fraction=wm_fraction,
                    seed=seed_val,
                )
                args.train_data_dir = clean_dir         # folder1 in DualFolderDataset
                args.train_data_dir_2 = watermarked_dir # folder2
                args.use_dual_dataset = True
                args.files_override = (clean_files, watermarked_files)
        else:
            # Single-folder mode. Prefer whichever source is actually set.
            args.train_data_dir = watermarked_dir or clean_dir
            args.train_data_dir_2 = None
            args.use_dual_dataset = False
        args.dual_shuffle = True  # deterministic seeded shuffle — never take filename order
        args.max_train_samples_2 = 0  # kept for SD CLI compatibility; unused by DualFolderDataset
        for key, value in training.items():
            if value is None:
                continue
            setattr(args, key, value)
        _apply_attr_dict(args, output_paths)

        for key in ("report_to", "tracker_project_name", "dataloader_num_workers", "validation_prompts", "seed"):
            v = _cfg_get(config, key)
            if v is not None:
                setattr(args, key, v)

        module.main_with_args(args)

        # After training, expose the chosen checkpoint as output_dir/model (relative symlink).
        # Whichever subdir is selected, mi_model_dir() and is_complete() both resolve via "model/".
        import os
        from pathlib import Path
        out_dir = Path(output_paths.get("output_dir") or "")
        if out_dir:
            use_best = bool(_cfg_get(config, "finetuning.use_best_model", False))
            src_name = "best_model_pipeline" if use_best else "final_pipeline"
            src_path = out_dir / src_name
            model_link = out_dir / "model"
            if src_path.exists() and not model_link.exists():
                os.symlink(src_name, str(model_link))
