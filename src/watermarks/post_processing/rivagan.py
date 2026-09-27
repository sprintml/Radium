import os
import re
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torchvision import transforms

from ..base import PostGenerationWatermark


class RivaGANWatermark(PostGenerationWatermark):
    def __init__(self, model_path: str = None, params: dict = None, device: str | None = None):
        """RivaGAN post-generation watermark wrapper.

        Args:
            model_path: Base path where ONNX models and assets live.
            params: dictionary of watermark parameters (expects a 'messages' key or message bits)
            device: If set to 'cuda' or 'cpu' to control ONNX provider selection.
        """
        self.model_path = model_path
        self.params = params or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        providers = []
        if self.device == "cuda":
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        providers.append("CPUExecutionProvider")

        encoder_path = os.path.join(self.model_path or "", "rivagan_encoder.onnx")
        decoder_path = os.path.join(self.model_path or "", "rivagan_decoder.onnx")

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"RivaGAN encoder ONNX not found at {encoder_path}")
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(f"RivaGAN decoder ONNX not found at {decoder_path}")

        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)

        # Parse messages from params into a torch tensor
        messages = self.params.get("messages", None)
        if messages is None:
            self.messages = None
        # handle string reprs of tensors, lists, numpy array, torch tensors
        elif isinstance(messages, str):
            # extract 0/1 digits from string like 'tensor([[0.,1.,... ]])'
            bits = [int(ch) for ch in re.findall(r"[01]", messages)]
            self.messages = torch.tensor(bits, dtype=torch.float32).unsqueeze(0)
        elif isinstance(messages, (list, tuple, np.ndarray)):
            self.messages = torch.tensor(np.array(messages).reshape(1, -1), dtype=torch.float32)
        elif isinstance(messages, torch.Tensor):
            self.messages = messages
        else:
            # fallback: try to coerce to numpy array
            try:
                arr = np.array(messages)
                self.messages = torch.tensor(arr.reshape(1, -1), dtype=torch.float32)
            except Exception:
                raise ValueError("Unsupported messages format in RivaGAN params")

    def apply(self, images: Any, args: dict = None, **kwargs) -> Any:
        """Apply RivaGAN-based watermark to image.

        # TODO: integrate trained RivaGAN model to apply watermark
        """
        if not isinstance(images, list):
            images = [images]

        # Convert list[PIL.Image] to a tensor batch (N, C, H, W)
        images_t = torch.stack([transforms.ToTensor()(img) for img in images])

        # Normalize to [-1,1] if in [0,1]
        if images_t.min() >= 0 and images_t.max() <= 1:
            images_t = (images_t - 0.5) * 2

        images_t = torch.clamp(images_t, -1.0, 1.0)

        # If model expects a channel for timesteps or frames, preserve the behavior in original code
        # Original code used an extra dimension (unsqueeze(2)); keep it but document as model-specific
        images_in = images_t.unsqueeze(2).detach().cpu().numpy()

        if self.messages is None:
            raise ValueError("RivaGANWindmark: messages missing in params")

        inputs = {
            "frame": images_in,
            "data": self.messages.detach().cpu().numpy(),
        }

        outputs = np.stack(self.encoder.run(None, inputs))

        wm_images_t = torch.from_numpy(outputs)
        wm_images_t = torch.clamp(wm_images_t, min=-1.0, max=1.0)
        wm_images_t = (wm_images_t / 2.0) + 0.5
        wm_images_t = wm_images_t.squeeze()

        to_pil = transforms.ToPILImage(mode="RGB")

        return [to_pil(wm_images_t)]

    def detect_features(self, images: Any) -> list[np.ndarray]:
        """Per-image raw decoder logits (K,) — same ONNX path detect() uses
        but without the ``> 0`` thresholding step, so the continuous signal is
        available to aggregation tests.
        """
        if not isinstance(images, list):
            images = [images]

        images_t = torch.stack([transforms.ToTensor()(img) for img in images])
        if images_t.min() >= 0 and images_t.max() <= 1:
            images_t = (images_t - 0.5) * 2
        images_t = torch.clamp(images_t, -1.0, 1.0)
        images_in = images_t.unsqueeze(2).detach().cpu().numpy()

        outputs = self.decoder.run(None, {"frame": images_in})
        if not outputs:
            raise RuntimeError("RivaGAN decoder returned no outputs")
        # outputs[0] is (N, K) (or (N, 1, K) per the ONNX export); flatten the
        # trailing dims so we get a clean (K,) logit vector per image.
        logits = np.asarray(outputs[0], dtype=np.float32)
        logits = logits.reshape(logits.shape[0], -1)
        return [logits[i] for i in range(logits.shape[0])]

    def planted_bits(self) -> np.ndarray:
        if self.messages is None:
            raise ValueError("RivaGAN: messages missing in params")
        return np.array(
            [int(x) for x in torch.flatten(self.messages.detach().cpu()).tolist()],
            dtype=np.int32,
        )

    def scores_from_features(self, features):
        bits = self.planted_bits()
        K = len(bits)
        return [
            float(((f[:K] > 0).astype(np.int32) == bits).mean()) for f in features
        ]

    def detect(self, images: Any, args: dict = None, **kwargs) -> float:
        feats = self.detect_features(images)
        scores = self.scores_from_features(feats)
        return scores[0] if len(scores) == 1 else scores
