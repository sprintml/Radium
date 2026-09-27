from __future__ import annotations

from typing import List, Sequence

import torch
from PIL import Image

from ..base import GenerationWatermark

from src.bitmark.architecture_wrapper import get_architecture, get_vae
from src.bitmark.detect_watermark import WatermarkInference, get_detector, detect
from src.bitmark.helper import get_watermark_scales


def tensor_to_pil(x):
    if isinstance(x, Image.Image):
        return x

    if torch.is_tensor(x):
        x = x.detach().cpu()

    if x.ndim == 4:
        x = x[0]

    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        pass  # already HWC
    elif x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = x.permute(1, 2, 0)
    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")

    if x.dtype != torch.uint8:
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        x = (x * 255).clamp(0, 255).to(torch.uint8)

    return Image.fromarray(x.numpy())


class BitMarkWatermark(GenerationWatermark):
    BATCH_EVAL = True

    def __init__(self, device=None, dic=None):
        super().__init__(device, dic)

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # By the time we get here, build_watermark() has already composed the
        # generation template + model base into dic. The Infinity architecture
        # block lives under dic.infinity (see configs/models/infinity_2b.yaml)
        # so it doesn't pollute watermark.params (and the path slug). Merge it
        # under the bitmark identity params here so existing args.<key> reads
        # keep working.
        #
        # Detection-only paths (e.g. computing bitmark scores on clean SD1.4 /
        # SD2.1 images for a cross-arch reference) load no Infinity model
        # block, but the bitmark detector itself is always Infinity-based —
        # any image is encoded through the Infinity VAE into the token space
        # the detector operates on. Fall back to loading the canonical
        # Infinity config so `args.architecture` etc. are populated.
        from pathlib import Path
        from omegaconf import OmegaConf
        infinity_cfg = getattr(dic, "infinity", None)
        if infinity_cfg is None:
            fallback = (Path(__file__).resolve().parents[3]
                        / "configs" / "models" / "infinity_2b.yaml")
            infinity_cfg = OmegaConf.load(fallback).get("infinity")
        self.args = OmegaConf.merge(infinity_cfg, dic.watermark.params)
        if getattr(self.args, "batch_size", None) is None:
            dataset_params = getattr(dic, "dataset_params", None)
            if dataset_params is not None and getattr(dataset_params, "batch_size", None) is not None:
                self.args.batch_size = int(dataset_params.batch_size)

        self.vae_wrapper = get_vae(self.args)
        self.arch_wrapper = get_architecture(self.args, self.vae_wrapper)
        self.wm_infer = WatermarkInference(self.args, self.vae_wrapper)
        self.detector = get_detector(self.args)

        self.watermark_scales = get_watermark_scales(
            getattr(self.args, "watermark_scales", 0),
            getattr(self.vae_wrapper, "scale_schedule", []),
        )

    def _generate_watermarked(self, prompts: Sequence[str]) -> List[Image.Image]:
        images = []
        for prompt in prompts:
            img = self.arch_wrapper.gen_img(
                prompts=prompt,
                vae=self.vae_wrapper.vae,
                watermark_inference=self.wm_infer,
            )
            images.append(tensor_to_pil(img))
        return images

    def _generate_clean(self, prompts: Sequence[str]) -> List[Image.Image]:
        # BitMark biases the token logits by watermark_delta; delta=0 disables
        # the bias and the output is equivalent to plain Infinity generation.
        # Build a separate WatermarkInference with delta=0 once, lazily.
        if not hasattr(self, "_clean_wm_infer"):
            from omegaconf import OmegaConf
            clean_args = OmegaConf.merge(self.args, OmegaConf.create({"watermark_delta": 0}))
            self._clean_wm_infer = WatermarkInference(clean_args, self.vae_wrapper)
        images = []
        for prompt in prompts:
            img = self.arch_wrapper.gen_img(
                prompts=prompt,
                vae=self.vae_wrapper.vae,
                watermark_inference=self._clean_wm_infer,
            )
            images.append(tensor_to_pil(img))
        return images

    def detect(self, images: Sequence[Image.Image]) -> List[float]:
        scores: List[float] = []
        for img in images:
            metrics = detect(
                args=self.args,
                img_path=img,
                watermark_detector=self.detector,
                vae_wrapper=self.vae_wrapper,
                watermark_scales=self.watermark_scales,
                detect_on_each_scale=False,
            )
            scores.append(float(metrics["z_score"]))
        return scores


__all__ = ["BitMarkWatermark"]
