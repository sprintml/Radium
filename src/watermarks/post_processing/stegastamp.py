import os
import re

import numpy as np
import onnxruntime as ort
import torch
from torchvision import transforms

from ..base import PostGenerationWatermark


class StegaStampWatermark(PostGenerationWatermark):
    def __init__(self, model_path: str = None, params: dict = None, device: str | None = None):
        """StegaStamp post-generation watermark wrapper.

        Args:
            model_path: Base path where ONNX models and assets live.
            params: dictionary of watermark parameters (expects a 'messages' key or message bits)
            device: If set to 'cuda' or 'cpu' to control ONNX provider selection.
        """
        self.model_path = model_path
        self.params = params or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._parse_messages()

        providers = []
        if self.device == "cuda":
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        providers.append("CPUExecutionProvider")

        encoder_path = os.path.join(self.model_path or "", "stega_stamp.onnx")

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"StegaStamp encoder not found at {encoder_path}")

        self.model = ort.InferenceSession(f"{encoder_path}", providers=providers)
        self.resize_down = transforms.Resize((400, 400))

    def _parse_messages(self):
        messages = self.params.get("messages", None)
        if messages is None:
            self.messages = None
        elif isinstance(messages, str):
            bits = [int(ch) for ch in re.findall(r"[01]", messages)]
            self.messages = torch.tensor(bits, dtype=torch.float32).unsqueeze(0)
        elif isinstance(messages, (list, tuple, np.ndarray)):
            self.messages = torch.tensor(np.array(messages).reshape(1, -1), dtype=torch.float32)
        elif isinstance(messages, torch.Tensor):
            self.messages = messages
        else:
            try:
                arr = np.array(messages)
                self.messages = torch.tensor(arr.reshape(1, -1), dtype=torch.float32)
            except Exception:
                raise ValueError("Unsupported messages format in StegaStamp params")

    def apply(self, images):
        messages = self.messages

        if not isinstance(images, list):
            images = [images]
            original_size = images[0].size[0]

        self.resize_up = transforms.Resize((original_size, original_size))

        images = torch.stack([transforms.ToTensor()(img) for img in images])
        inputs = {
            "image": self.resize_down(images).permute(0, 2, 3, 1).detach().cpu().float().numpy(),
            "secret": messages.detach().cpu().numpy(),
        }
        wm_images = np.stack(self.model.run(None, inputs)[0])
        wm_images = torch.from_numpy(wm_images)
        wm_images = wm_images.permute(0, 3, 1, 2)
        wm_images = self.resize_up(wm_images)
        to_pil = transforms.ToPILImage()
        return [to_pil(wm_images[i]) for i in range(wm_images.size(0))]

    def detect_features(self, images):
        """Run the ONNX extractor once and return per-image (100,) probability vectors."""
        if not isinstance(images, list):
            images = [images]
        images_t = torch.stack([transforms.ToTensor()(img) for img in images])
        inputs = {
            "image": self.resize_down(images_t).permute(0, 2, 3, 1).detach().cpu().float().numpy(),
            "secret": np.zeros((len(images), 100), dtype=np.float32),
        }
        message = self.model.run(None, inputs)[2].astype(np.float32)
        return [message[i] for i in range(message.shape[0])]

    def planted_bits(self):
        if self.messages is None:
            raise ValueError("StegaStamp: messages missing in params")
        return np.array(
            [int(x) for x in torch.flatten(self.messages.detach().cpu()).tolist()],
            dtype=np.int32,
        )

    def scores_from_features(self, features):
        bits = self.planted_bits()
        K = len(bits)
        return [
            float(((f[:K] > 0.5).astype(np.int32) == bits).mean()) for f in features
        ]

    def detect(self, images):
        feats = self.detect_features(images)
        scores = self.scores_from_features(feats)
        # Preserve historical scalar return: detect() was called per-image,
        # so a singleton list collapses to a scalar.
        return scores[0] if len(scores) == 1 else scores
