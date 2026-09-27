from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence
from typing import Any


import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from ..base import PostGenerationWatermark

HERE = Path(__file__).resolve().parent
SIREN_ROOT = (HERE.parents[1] / "SIREN_deps").resolve()
if str(SIREN_ROOT) not in sys.path:
    sys.path.insert(0, str(SIREN_ROOT))

print(HERE, SIREN_ROOT)  # sanity check for correct path resolution

from lib.attenuations import JND  # noqa: E402
from lib.models import HiddenDecoder, HiddenEncoder  # noqa: E402


DEFAULT_MODELS_ROOT = Path.home() / "data_dir/models/siren"
DEFAULT_DATA_ROOT = Path.home() / "data_dir/data"

UNNORMALIZE_IMAGENET = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)
NORMALIZE_IMAGENET = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class SIRENWatermark(PostGenerationWatermark):
    """SIREN post-processing watermark wrapper.

    Expected checkpoint layout under models_root (configurable):
    - meta/hidden_encoder.pth
    - meta/hidden_decoder.pth

    You can override each checkpoint path via params.
    """

    def __init__(self, model_path: str | None = None, params: dict | None = None, device: str | None = None):
        self.params = params or {}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        default_models_root = Path(model_path) if model_path else DEFAULT_MODELS_ROOT
        self.models_root = Path(self.params.get("models_root", default_models_root))
        self.data_root = Path(self.params.get("data_root", DEFAULT_DATA_ROOT))

        self.resolution = int(self.params.get("resolution", 512))
        self.strength = float(self.params.get("strength", 1.5))
        self.checkpoint_set = str(self.params.get("checkpoint_set")).lower()

        self.meta_encoder_checkpoint = Path(
            self.params.get("meta_encoder_checkpoint", self.models_root / "meta" / "hidden_encoder.pth")
        )
        self.meta_decoder_checkpoint = Path(
            self.params.get("meta_decoder_checkpoint", self.models_root / "meta" / "hidden_decoder.pth")
        )

        self.encoder_checkpoint = Path(
            self.params.get("encoder_checkpoint", self._get_encoder_checkpoint(self.checkpoint_set))
        )
        self.decoder_checkpoint = Path(
            self.params.get("decoder_checkpoint", self._get_decoder_checkpoint(self.checkpoint_set))
        )

        self._to_tensor = transforms.ToTensor()
        self._to_pil = transforms.ToPILImage()

        self.encoder = HiddenEncoder(num_blocks=4, num_bits=48, channels=64)
        self.decoder = HiddenDecoder(num_blocks=8, num_bits=48, channels=64)
        self.attenuation = JND(preprocess=UNNORMALIZE_IMAGENET).to(self.device)
        self.attenuation.requires_grad_(False)

        self._load_checkpoints()

    def _get_encoder_checkpoint(self, checkpoint_set: str) -> Path:
        if checkpoint_set == "meta":
            return self.meta_encoder_checkpoint
        return self.encoder_checkpoint

    def _get_decoder_checkpoint(self, checkpoint_set: str) -> Path:
        if checkpoint_set == "meta":
            return self.meta_decoder_checkpoint
        return self.decoder_checkpoint

    def _default_msg(self) -> torch.Tensor:
        msg_bits = self.params.get("msg_bits")
        if msg_bits is None:
            return torch.randint(0, 2, (1, 48), dtype=torch.float32)

        if isinstance(msg_bits, str):
            bits = [float(ch) for ch in msg_bits if ch in ("0", "1")]
            if len(bits) != 48:
                raise ValueError("msg_bits string must contain exactly 48 bits")
            return torch.tensor(bits, dtype=torch.float32).unsqueeze(0)

        msg_tensor = torch.tensor(msg_bits, dtype=torch.float32)
        if msg_tensor.ndim == 1:
            msg_tensor = msg_tensor.unsqueeze(0)
        if msg_tensor.shape[-1] != 48:
            raise ValueError("msg_bits must contain 48 elements")
        return msg_tensor

    def _load_checkpoints(self) -> None:
        if not self.encoder_checkpoint.exists():
            raise FileNotFoundError(f"SIREN encoder checkpoint not found: {self.encoder_checkpoint}")
        if not self.decoder_checkpoint.exists():
            raise FileNotFoundError(f"SIREN decoder checkpoint not found: {self.decoder_checkpoint}")

        enc_state = torch.load(self.encoder_checkpoint, map_location="cpu")
        if "msg" in enc_state:
            self.encoder.get_msg(enc_state["msg"])
            self.encoder.load_state_dict(enc_state)
        else:
            self.encoder.load_state_dict(enc_state)
            self.encoder.get_msg(self._default_msg())
        self.encoder = self.encoder.to(self.device)
        self.encoder.requires_grad_(False)

        dec_state = torch.load(self.decoder_checkpoint, map_location="cpu")
        if "center" in dec_state:
            self.decoder.get_center(dec_state["center"])
            self.decoder.load_state_dict(dec_state)
        else:
            center = torch.zeros((1, 48), dtype=torch.float32)
            self.decoder.get_center(center)
            self.decoder.load_state_dict(dec_state, strict=False)
        self.decoder = self.decoder.to(self.device)
        self.decoder.requires_grad_(False)

    def _prepare_batch(self, images: Sequence[Image.Image]) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        prepared = []
        original_sizes = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError("SIRENWatermark.apply expects PIL images")
            image = image.convert("RGB")
            original_sizes.append(image.size)
            resized = image.resize((self.resolution, self.resolution), Image.BICUBIC)
            prepared.append(self._to_tensor(resized))
        return torch.stack(prepared).to(self.device), original_sizes

    def apply(self, images: Image.Image | Iterable[Image.Image]):
        if isinstance(images, Image.Image):
            image_list = [images]
        else:
            image_list = list(images)
        if not image_list:
            return []

        with torch.no_grad():
            batch, original_sizes = self._prepare_batch(image_list)
            normalized = NORMALIZE_IMAGENET(batch)

            msg = self.encoder.msg
            if msg.ndim == 1:
                msg = msg.unsqueeze(0)
            msg = (msg * 2 - 1).to(self.device)
            msg = msg.expand(normalized.shape[0], -1)

            deltas_w = self.encoder(normalized, msg)
            mask = self.attenuation.heatmaps(normalized)
            mask[:, :, 0, :] = 0
            mask[:, :, -1, :] = 0
            mask[:, :, :, 0] = 0
            mask[:, :, :, -1] = 0
            watermarked = normalized + self.strength * (deltas_w * mask)
            watermarked = torch.clamp(UNNORMALIZE_IMAGENET(watermarked), 0, 1)

        outputs: List[Image.Image] = []
        for idx, tensor_image in enumerate(watermarked):
            pil_image = self._to_pil(tensor_image.cpu())
            if pil_image.size != original_sizes[idx]:
                pil_image = pil_image.resize(original_sizes[idx], Image.BICUBIC)
            outputs.append(pil_image)
        return outputs

    def detect(self, images: Image.Image | Iterable[Image.Image]):
        if isinstance(images, Image.Image):
            image_list = [images]
        else:
            image_list = list(images)
        if not image_list:
            return []

        with torch.no_grad():
            batch, _ = self._prepare_batch(image_list)
            normalized = NORMALIZE_IMAGENET(batch)
            pred = self.decoder(normalized)
            center = self.decoder.center
            values = torch.sqrt(torch.norm(pred - center, p=2, dim=1) ** 2 + 1) - 1
        return float(-1* values.item()) #negative such that higher score (lower distance) is stronger detection


__all__ = ["SIRENWatermark", "DEFAULT_MODELS_ROOT", "DEFAULT_DATA_ROOT"]
