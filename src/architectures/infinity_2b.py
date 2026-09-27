"""Infinity 2B backend.

Owns ``InfinityCleanGenerator`` — Infinity's clean path is BitMark with
delta=0 (same generator and logits processor; bias zeroed so sampling is
equivalent to plain Infinity).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence

from PIL import Image

from src.architectures.base import CleanGenerator, ModelBackend, get_backend, register


class InfinityCleanGenerator(CleanGenerator):
    """Infinity clean generator: BitMark architecture with delta=0."""

    def __init__(self, config):
        from omegaconf import OmegaConf

        from src.bitmark.architecture_wrapper import get_architecture, get_vae
        from src.bitmark.detect_watermark import WatermarkInference

        infinity_cfg = config.get("infinity") if hasattr(config, "get") else None
        if infinity_cfg is None:
            raise ValueError(
                "Infinity clean generation requires an 'infinity' config block"
            )
        # WatermarkInference needs BitMark's params (watermark_scales,
        # watermark_method, set, ...) on `args` to boot; they live under
        # config.watermark.params (BitMark base from
        # configs/models/infinity_2b.yaml). watermark_delta is forced to 0
        # here regardless of the active watermark config.
        model_cfg = config.get("model", {}) or {}
        active_model_path = model_cfg.get("model_path") or infinity_cfg.get("model_path")
        wm_params = (config.get("watermark", {}) or {}).get("params")
        if wm_params is None:
            raise ValueError(
                "Infinity clean generation requires watermark.params (BitMark base) "
                "to be present in the config — provided by configs/models/infinity_2b.yaml."
            )
        # The planner injects model.model_path as a directory (Mi output dir).
        # Infinity's load_infinity wants a file — resolve via the backend.
        if active_model_path:
            active_model_path = get_backend("infinity_2b").resolve_checkpoint_path(active_model_path)
        overrides = OmegaConf.create({
            "watermark_delta": 0,
            "model_path": active_model_path,
        })
        dataset_params = config.get("dataset_params", {}) or {}
        if dataset_params.get("batch_size") is not None:
            overrides.batch_size = int(dataset_params.get("batch_size"))

        self._args = OmegaConf.merge(infinity_cfg, wm_params, overrides)
        self._vae = get_vae(self._args)
        self._arch = get_architecture(self._args, self._vae)
        self._wm_infer = WatermarkInference(self._args, self._vae)

    def generate(self, prompts: Sequence[str]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for prompt in prompts:
            img = self._arch.gen_img(
                prompts=prompt,
                vae=self._vae.vae,
                watermark_inference=self._wm_infer,
            )
            images.append(_tensor_to_pil(img))
        return images


def _tensor_to_pil(x):
    import torch

    if isinstance(x, Image.Image):
        return x
    if torch.is_tensor(x):
        x = x.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        pass  # HWC
    elif x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = x.permute(1, 2, 0)
    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")
    if x.dtype != torch.uint8:
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        x = (x * 255).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(x.numpy())


class Infinity2BBackend(ModelBackend):
    name = "infinity_2b"
    aliases = ("infinity",)

    def build_clean_generator(self, config):
        return InfinityCleanGenerator(config)

    def build_finetuning_runner(self):
        from src.finetuning.infinity.runner import InfinityRunner
        return InfinityRunner()

    def resolve_checkpoint_path(self, path) -> str:
        # Infinity's load_infinity / train.py:rush_resume both want a concrete
        # .pth file. When handed a dir, prefer the runner's selection symlink
        # (ar-ckpt-selected.pth -> best or last). Fall back to ar-ckpt-last,
        # ar-ckpt-best, then any ar-ckpt-*.pth picked by ep/iter.
        p = Path(path)
        if p.is_file():
            return str(p)
        if not p.is_dir():
            raise FileNotFoundError(f"Infinity checkpoint not found: {p}")
        for preferred in ("ar-ckpt-selected.pth", "ar-ckpt-last.pth", "ar-ckpt-best.pth"):
            cand = p / preferred
            if cand.exists():
                return str(cand)
        cands = [c for c in p.iterdir() if c.is_file() and c.name.startswith("ar-ckpt-") and c.suffix == ".pth"]
        if not cands:
            raise FileNotFoundError(f"No ar-ckpt-*.pth in {p} — did finetuning sync to bed?")
        def key(c: Path) -> tuple[int, int]:
            m = re.search(r"ep(\d+)-iter(\d+)", c.name)
            return (int(m[1]), int(m[2])) if m else (0, 0)
        return str(max(cands, key=key))


register(Infinity2BBackend())
