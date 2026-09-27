"""ModelBackend ABC: one place to add a new model architecture.

A backend bundles every per-architecture extension point the pipeline needs
(clean image generation, finetuning runner construction). Adding a new
architecture means creating one ``ModelBackend`` subclass — no edits to
any dispatch site elsewhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, List, Sequence

from PIL import Image

if TYPE_CHECKING:
    from src.finetuning.runner import FinetuningRunner


class CleanGenerator(ABC):
    """Loads a generator once; ``generate`` is called per batch.

    In ``clean_generate`` mode the watermark is irrelevant — the generator
    is just the underlying diffusion / autoregressive model. Concrete
    subclasses live in each architecture module under ``src/architectures/``.
    """

    @abstractmethod
    def generate(self, prompts: Sequence[str]) -> List[Image.Image]: ...


class ModelBackend(ABC):
    """One concrete subclass per architecture (sd21, infinity_2b, ...).

    Subclasses must set ``name`` and ``aliases`` as class attributes and
    implement ``build_clean_generator`` / ``build_finetuning_runner``.
    """

    name: str = ""
    aliases: Sequence[str] = ()

    @abstractmethod
    def build_clean_generator(self, config) -> CleanGenerator:
        """Construct the clean (no-watermark) image generator for this architecture."""

    @abstractmethod
    def build_finetuning_runner(self) -> "FinetuningRunner":
        """Construct the finetuning runner for this architecture."""

    def resolve_checkpoint_path(self, path) -> str:
        """Normalize a checkpoint path for this backend's loaders.

        The pipeline planner injects ``model.model_path`` / ``model.pretrained_model_name_or_path``
        as the previous step's Mi *directory*. Most loaders (HF ``from_pretrained``)
        accept directories directly, so the default is passthrough. Override
        when a loader expects a specific shape — e.g. Infinity needs a concrete
        ``.pth`` file, so it picks the latest checkpoint inside the dir.
        """
        return str(path)


_REGISTRY: dict[str, ModelBackend] = {}


def register(backend: ModelBackend) -> None:
    """Add a backend to the registry under its ``name`` and every alias.

    Aliases are case-insensitive. Duplicate registrations overwrite — last
    import wins, which is fine because each backend lives in exactly one file.
    """
    if not backend.name:
        raise ValueError(f"{type(backend).__name__}.name is required")
    keys = {backend.name.lower(), *(a.lower() for a in backend.aliases)}
    for key in keys:
        _REGISTRY[key] = backend


def get_backend(name: str | None) -> ModelBackend:
    """Look up a backend by name or alias (case-insensitive)."""
    if not name:
        raise ValueError("model type is required to look up a backend")
    backend = _REGISTRY.get(str(name).lower())
    if backend is None:
        supported = ", ".join(sorted(_REGISTRY))
        raise NotImplementedError(
            f"No ModelBackend registered for '{name}'. Supported: {supported}. "
            f"Add a new entry under src/architectures/."
        )
    return backend


def registered_backends() -> Iterable[tuple[str, ModelBackend]]:
    """Iterate (alias, backend) pairs — each backend appears once per alias."""
    return _REGISTRY.items()


def build_clean_generator(config) -> CleanGenerator:
    """Dispatch clean-generator construction by ``mi_model_type`` / ``m1_model_type``."""
    model_type = (
        config.get("mi_model_type") if hasattr(config, "get") else None
    ) or (config.get("m1_model_type") if hasattr(config, "get") else None)
    if not model_type:
        raise ValueError(
            "clean generation requires m1_model_type or mi_model_type in config"
        )
    return get_backend(str(model_type)).build_clean_generator(config)
