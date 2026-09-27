"""Base class and shared utilities for finetuning runners."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import OmegaConf

from src.utils.commonargs import cfg_get, load_config

logger = logging.getLogger(__name__)


def load_finetune_config(
    config_path: str,
    watermark_config_path: str | None = None,
    model_type: str | None = None,
):
    """Load a run config and merge the per-model finetuning base under it.

    The model config (``configs/models/<type>.yaml``, merged in by
    ``compose_generation_config`` inside ``load_config``) declares
    ``finetune_base`` — the YAML carrying training hyperparameters for that
    architecture. A base may itself declare ``finetune_base`` to extend
    another base (e.g. ``stablediffusion/sd3.yaml`` chaining
    ``stablediffusion/common.yaml``); the chain is walked recursively and
    each layer is merged so later configs override earlier ones.
    """
    run_cfg = load_config(
        config_path,
        watermark_config_path=watermark_config_path,
        model_type=model_type,
    )
    return _merge_finetune_chain(run_cfg, visited=set())


def _merge_finetune_chain(cfg, visited: set[str]):
    base_path = cfg.get("finetune_base") if hasattr(cfg, "get") else None
    if not base_path:
        return cfg
    base_str = str(base_path)
    if base_str in visited:
        return cfg
    visited.add(base_str)
    path = Path(base_str)
    if not path.exists():
        raise FileNotFoundError(
            f"finetune_base={base_str!r} declared in config but file is missing. "
            f"Either create it or repoint at an existing file. Silently skipping "
            f"would drop training defaults and produce opaque downstream failures."
        )
    base_cfg = _merge_finetune_chain(OmegaConf.load(str(path)), visited)
    return OmegaConf.merge(base_cfg, cfg)


def save_run_config(config, output_dir: str | None, filename: str = "run_config.yaml") -> str | None:
    if not output_dir:
        return None
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, str(out_path / filename))
    return str(out_path / filename)


def _apply_attr_dict(args, values: dict[str, object]) -> None:
    for key, value in values.items():
        setattr(args, key, value)


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def assert_input_dir(label: str, directory: str | None, *, min_files: int = 1) -> None:
    """Fail fast if a runner input directory is missing or empty.

    Raises with the upstream stage in the message so the failure points at
    the actual cause instead of crashing inside model load.
    """
    if not directory:
        raise FileNotFoundError(
            f"{label} not configured. Upstream stage likely failed to write — "
            f"check the parent generation/finetuning job."
        )
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label}: {directory} does not exist. "
            f"Upstream stage failed to write — check the parent job."
        )
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    if len(files) < min_files:
        raise FileNotFoundError(
            f"{label}: {directory} has {len(files)} image(s), need at least {min_files}. "
            f"Upstream stage produced an empty/partial directory."
        )


class FinetuningRunner(ABC):
    """Base class for all finetuning backends.

    To add a new backend, create ``src/architectures/<name>.py`` with a
    ``ModelBackend`` subclass whose ``build_finetuning_runner()`` returns an
    instance of a ``FinetuningRunner`` subclass implementing ``_run`` and
    ``is_complete``. The architecture's registry lookup wires it into the
    rest of the pipeline — no edits to dispatch sites elsewhere.
    """

    def execute(self, config, output_paths: dict[str, str | None]) -> None:
        """Prepare output directory, snapshot config, then delegate to _run."""
        out_dir = output_paths.get("output_dir")
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            saved = save_run_config(config, out_dir)
            if saved:
                logger.info("Saved config snapshot to %s", saved)
        self._run(config, output_paths)

    @abstractmethod
    def _run(self, config, output_paths: dict[str, str | None]) -> None: ...

    @abstractmethod
    def is_complete(self, output_dir: str) -> bool:
        """Return True if a trained model already exists at output_dir."""
        ...
