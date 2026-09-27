"""SD1.4 backend — same clean generator and finetuning module as SD2.1."""

from __future__ import annotations

from src.architectures.base import ModelBackend, register
from src.architectures.sd21 import SDCleanGenerator


class SD14Backend(ModelBackend):
    name = "sd14"
    aliases = ("stablediffusion14", "stable-diffusion-1-4", "stable-diffusion-v1-4")

    def build_clean_generator(self, config):
        return SDCleanGenerator(config)

    def build_finetuning_runner(self):
        from src.finetuning.sd_runner import SDRunner

        def _load():
            from src.finetuning.sd_21_base import finetunable_stable_diffusion
            return finetunable_stable_diffusion

        return SDRunner("sd14", _load)


register(SD14Backend())
