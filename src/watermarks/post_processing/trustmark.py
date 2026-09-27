from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from trustmark import TrustMark

from ..base import PostGenerationWatermark


class TrustMarkWatermark(PostGenerationWatermark):
    def __init__(self, params: dict = None, args=None, **kwargs):
        self.params = params or {}
        self.model = TrustMark(verbose=True, model_type="Q", use_ECC=False, loadBBoxDetector=False)
        self.bitmessage = self.params.get("bitmessage", None)

        print(f"Initializing TrustMarkWatermark with bitmessage: {self.bitmessage}")
        print(self.bitmessage)

    def apply(self, images) -> Any:
        """Apply TrustMark watermarking algorithm.

        # TODO: implement embedding and metadata handling
        """
        wmarked_image = self.model.encode(images, self.bitmessage, MODE="binary")
        return [wmarked_image]

    @torch.no_grad()
    def _decode_logits(self, image) -> np.ndarray:
        """Run the TrustMark decoder on one image and return raw pre-threshold logits (K,)."""
        processed = self.model.get_the_image_for_processing(image)
        resized = processed.resize(
            (self.model.model_resolution_dec, self.model.model_resolution_dec),
            Image.BILINEAR,
        )
        stego = (
            transforms.ToTensor()(resized).unsqueeze(0).to(self.model.decoder.device) * 2.0 - 1.0
        )
        logits = self.model.decoder.decoder(stego)  # (1, secret_len)
        return logits[0].detach().cpu().numpy().astype(np.float32)

    def detect_features(self, images):
        """Per-image raw decoder logits (K,) — same path the library uses but
        without the ``> 0`` thresholding step, so probabilities are recoverable
        and aggregation tests have access to the continuous signal.
        """
        if not isinstance(images, list):
            images = [images]
        return [self._decode_logits(img) for img in images]

    def planted_bits(self):
        if self.bitmessage is None:
            raise ValueError("TrustMark: bitmessage missing in params")
        return np.array([int(c) for c in str(self.bitmessage)], dtype=np.int32)

    def scores_from_features(self, features):
        bits = self.planted_bits()
        K = len(bits)
        return [
            float(((f[:K] > 0).astype(np.int32) == bits).mean()) for f in features
        ]

    def detect(self, images):
        feats = self.detect_features(images)
        scores = self.scores_from_features(feats)
        return scores[0] if len(scores) == 1 else scores
