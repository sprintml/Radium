"""Per-architecture backend modules.

To add a new model architecture:
  1. Create ``src/architectures/<name>.py`` exporting a single ``ModelBackend``
     subclass (with its concrete ``CleanGenerator``) that registers itself at
     module import.
  2. Add ``from src.architectures import <name>`` below.

Everything that dispatches by ``m1_model_type`` / ``mi_model_type``
(generation, finetuning, future stages) routes through ``get_backend(name)``
or the convenience wrapper ``build_clean_generator(config)`` — no per-model
branching elsewhere.
"""

from __future__ import annotations

from src.architectures.base import (
    CleanGenerator,
    ModelBackend,
    build_clean_generator,
    get_backend,
    register,
    registered_backends,
)

# Side-effect imports: each module's bottom-level ``register(...)`` call
# wires the backend into the registry. Order is irrelevant.
from src.architectures import sd14, sd21, sd3, infinity_2b  # noqa: F401

__all__ = [
    "CleanGenerator",
    "ModelBackend",
    "build_clean_generator",
    "get_backend",
    "register",
    "registered_backends",
]
