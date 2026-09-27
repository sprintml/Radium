from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image


class PostGenerationWatermark(ABC):
    """Base class for post-generation watermarks (applied to existing images)."""

    supports_batch_evaluate = False

    def __init__(self, device: torch.device | None = None, params=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.params = params

    @abstractmethod
    def apply(self, images: torch.Tensor | Image.Image) -> torch.Tensor | Image.Image:
        """Embed watermark into image(s). Returns same type as input."""

    @abstractmethod
    def detect(self, images: torch.Tensor | Image.Image) -> float:
        """Score a single image for watermark presence. Returns a float."""

    def detect_features(self, images) -> list[np.ndarray]:
        """Per-image carrier-aligned feature vector for aggregation tests.

        Override in watermarks that expose a meaningful per-bit / per-coefficient
        feature (e.g. raw extractor logits or Fourier-ring projections). Each
        returned vector must have the same shape across calls so the detection
        stage can stack them into a single (N, K) cache.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement detect_features"
        )

    def scores_from_features(self, features: list[np.ndarray]) -> list[float]:
        """Per-image bit accuracy from features. Override per watermark — each
        inlines its own decode threshold in native space.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement scores_from_features"
        )


class GenerationWatermark(ABC):
    """Base class for in-generation watermarks (injected during diffusion).

    Subclasses must implement ``_generate_watermarked``, ``_generate_clean``,
    and ``detect``. The ``generate`` method dispatches based on ``mode``
    so subclasses cannot accidentally ignore the clean-generate path.
    """

    supports_batch_evaluate = True

    def __init__(self, device: torch.device | None = None, dic=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dic = dic
        wm_cfg = dic.get("watermark", {}) if dic else {}
        self.mode = (wm_cfg or {}).get("mode", "generate")

    def resolve_m1_pipeline_path(self) -> str | None:
        """Return the model path the watermark's M1 pipeline should load.

        Cross-arch detection (mi != M1) needs the M1 base pipeline rather than
        the chain-injected mi-finetuned ``model.*``. compose_generation_config
        promotes M1's base block to ``m1_model:`` precisely for this — prefer
        it when present and m1 differs from mi. Otherwise fall back to
        ``dic.model.model_path`` (same-arch / detection_m1).
        """
        dic = self.dic or {}
        model_cfg = dic.get("model", {}) or {}
        m1_model_cfg = dic.get("m1_model", {}) or {}
        m1_type = dic.get("m1_model_type")
        mi_type = dic.get("mi_model_type") or m1_type
        if m1_model_cfg and m1_type and mi_type and m1_type != mi_type:
            return (m1_model_cfg.get("model_path") or m1_model_cfg.get("model_id")
                    or model_cfg.get("model_path") or model_cfg.get("model_id"))
        return model_cfg.get("model_path") or model_cfg.get("model_id")

    def generate(self, prompts: list[str]) -> list[Image.Image]:
        """Dispatch to watermarked or clean generation based on self.mode."""
        if self.mode == "clean_generate":
            return self._generate_clean(prompts)
        if self.mode == "generate":
            return self._generate_watermarked(prompts)
        raise ValueError(f"Unknown watermark mode '{self.mode}'. Expected 'generate' or 'clean_generate'.")

    @abstractmethod
    def _generate_watermarked(self, prompts: list[str]) -> list[Image.Image]:
        """Generate images with the watermark embedded."""

    @abstractmethod
    def _generate_clean(self, prompts: list[str]) -> list[Image.Image]:
        """Generate images without watermark injection."""

    @abstractmethod
    def detect(self, images: list[Image.Image]) -> list[float]:
        """Score a batch of images for watermark presence."""

    def detect_features(self, images) -> list[np.ndarray]:
        """Per-image carrier-aligned feature vector for aggregation tests.

        Override in watermarks that expose a meaningful per-bit / per-coefficient
        feature. Each returned vector must have the same shape across calls so
        the detection stage can stack them into a single (N, K) cache.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement detect_features"
        )

    def scores_from_features(self, features: list[np.ndarray]) -> list[float]:
        """Per-image bit accuracy from features. Override per watermark — each
        inlines its own decode threshold in native space.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement scores_from_features"
        )
