"""SD2.1 backend.

Owns ``SDCleanGenerator`` — the SD-family clean image generator shared by
sd14 and sd3 (they import it from here).
"""

from __future__ import annotations

from typing import List, Sequence

from PIL import Image

from src.architectures.base import CleanGenerator, ModelBackend, register


class SDCleanGenerator(CleanGenerator):
    """Stable Diffusion family clean generator (sd14 / sd21 )."""

    def __init__(self, config):
        import torch
        from diffusers import StableDiffusionPipeline

        model_cfg = config.get("model", {}) or {}
        model_path = model_cfg.get("model_path") or model_cfg.get("model_id")
        if not model_path:
            raise ValueError("model.model_path (or model.model_id) must be set")

        revision = model_cfg.get("revision")
        kwargs = {"torch_dtype": torch.float16}
        if revision:
            kwargs["revision"] = revision

        self.pipe = StableDiffusionPipeline.from_pretrained(model_path, **kwargs).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.pipe.set_progress_bar_config(disable=True, leave=False)
        self._dataset_params = config.get("dataset_params", {}) or {}

    def generate(self, prompts: Sequence[str]) -> List[Image.Image]:
        import torch

        dp = self._dataset_params
        with torch.no_grad():
            outputs = self.pipe(
                list(prompts),
                num_images_per_prompt=int(dp.get("num_images_per_prompt", 1)),
                guidance_scale=float(dp.get("guidance_scale", 7.5)),
                num_inference_steps=int(dp.get("num_inference_steps", 50)),
                height=int(dp.get("image_length", 512)),
                width=int(dp.get("image_length", 512)),
            )
        return list(outputs.images)


class SD21Backend(ModelBackend):
    name = "sd21"
    aliases = ("stablediffusion", "stable-diffusion-2-1")

    def build_clean_generator(self, config):
        return SDCleanGenerator(config)

    def build_finetuning_runner(self):
        from src.finetuning.sd_runner import SDRunner

        def _load():
            from src.finetuning.sd_21_base import finetunable_stable_diffusion
            return finetunable_stable_diffusion

        return SDRunner("stablediffusion", _load)


register(SD21Backend())
