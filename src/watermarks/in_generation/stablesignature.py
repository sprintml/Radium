import os
import sys
import json
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from sklearn import metrics
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from omegaconf import OmegaConf

from src.model_backends.ldm.sd_21_base.utils_model import load_model_from_config
from ..base import GenerationWatermark
from src.utils.optim_utils import transform_img


class StableSignatureWatermark(GenerationWatermark):
    def __init__(self, device="cuda", dic=None):
        sys.path.append('src')
        sys.path.append('src/model_backends')
        
        super().__init__(device, dic)
        self.model_key = dic.watermark.params.model_key
        self.detector_pth = dic.watermark.detector_pth if hasattr(dic.watermark, 'detector_pth') else "models/dec_48b_whit.torchscript.pt"
        
        # Load the message extractor model
        print(f"Loading message extractor from {self.detector_pth}...", flush=True)
        self.msg_extractor = torch.jit.load(self.detector_pth).to(self.device)
        
        # Load the Stable Diffusion pipeline. For cross-arch detection
        # resolve_m1_pipeline_path falls back to M1's base path so we don't
        # try to load a non-SD mi-finetuned model.
        model_path = self.resolve_m1_pipeline_path() or "stabilityai/stable-diffusion-2-1-base"
        print(f">>> Initializing Stable Diffusion pipeline from {model_path}...", flush=True)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()  # Enable slicing for optimized memory usage
        # Disable diffusers' internal tqdm progress bars for inference steps
        self.pipe.set_progress_bar_config(disable=True, leave=False)
        
        if self.mode != "clean_generate":
            print("Loading custom fine-tuned decoder for StableSignature watermarking...", flush=True)
            self._load_decoder()
        
        if hasattr(self.dic.datasets, 'output_dir'):
            os.makedirs(f"{self.dic.datasets.output_dir}", exist_ok=True)
        else:
            print("No output directory specified in configuration.", flush=True)

    def _load_decoder(self):

        ldm_conf = OmegaConf.load(f"{self.dic.watermark.ldm_config_path}")
        # instantiate_from_config builds a fresh LatentDiffusion whose
        # nn.Module init (kaiming_uniform_, etc.) drains the global RNG by
        # ~MB before load_state_dict overwrites the weights. Fork so the
        # generation loop's noise stream matches the clean run, which never
        # constructs this model. Forking torch CPU+CUDA only — load_state_dict
        # and .cuda() don't touch numpy/python random.
        with torch.random.fork_rng():
            ldm_ae = load_model_from_config(ldm_conf, self.dic.watermark.ldm_ckpt)
        ldm_aef = ldm_ae.first_stage_model
        ldm_aef.eval()

        # loading the fine-tuned decoder weights
        state_dict = torch.load(self.dic.watermark.decoder_weights, map_location="cpu")
        unexpected_keys = ldm_aef.load_state_dict(state_dict, strict=False)
        print(unexpected_keys)
        print("you should check that the decoder keys are correctly matched")

        # Move decoder to device
        ldm_aef = ldm_aef.to(self.device).half()

        # loading the pipeline, and replacing the decode function of the pipe
        # Keep reference to ldm_aef so it stays in memory
        self.ldm_aef = ldm_aef
        self.pipe.vae.decode = (lambda x,  *args, **kwargs: self.ldm_aef.decode(x).unsqueeze(0))

    def _generate_watermarked(self, prompts):
        dataset_params = self.dic.dataset_params
        with torch.no_grad():
            outputs = self.pipe(
                prompts,
                num_images_per_prompt=dataset_params.num_images_per_prompt,
                guidance_scale=dataset_params.guidance_scale,
                num_inference_steps=dataset_params.num_inference_steps,
                height=dataset_params.image_length,
                width=dataset_params.image_length,
            )
        return list(outputs.images)

    def _generate_clean(self, prompts):
        dataset_params = self.dic.dataset_params
        with torch.no_grad():
            outputs = self.pipe(
                prompts,
                num_images_per_prompt=dataset_params.num_images_per_prompt,
                guidance_scale=dataset_params.guidance_scale,
                num_inference_steps=dataset_params.num_inference_steps,
                height=dataset_params.image_length,
                width=dataset_params.image_length,
            )
        return list(outputs.images)

    @torch.no_grad()
    def _extract_logits(self, img: Image.Image | torch.Tensor) -> np.ndarray:
        """Run msg_extractor once and return raw (K,) logits for one image."""
        if isinstance(img, Image.Image):
            img_tensor = self._transform_image(img).unsqueeze(0).to(self.device)
        else:
            img_tensor = img.unsqueeze(0) if img.dim() == 3 else img
            img_tensor = img_tensor.to(self.device)
        img_tensor = img_tensor.type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )
        msg = self.msg_extractor(img_tensor)  # (1, K)
        return msg.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def detect_features(self, images):
        """Per-image raw msg_extractor logits (K,) — the same path detect() uses
        but without the ``> 0`` thresholding step, so the continuous signal is
        retained for aggregation tests.
        """
        if not isinstance(images, list):
            images = [images]
        return [self._extract_logits(img) for img in images]

    def planted_bits(self) -> np.ndarray:
        return np.array(
            [int(c) for c in self.model_key if c in "01"], dtype=np.int32
        )

    def scores_from_features(self, features):
        bits = self.planted_bits()
        K = len(bits)
        return [
            float(((f[:K] > 0).astype(np.int32) == bits).mean()) for f in features
        ]

    def detect(self, images):
        """
        Evaluate and detect StableSignature watermarks in images.

        images: List[PIL.Image] or List[torch.Tensor]
        returns: List[float] - detection scores (bit accuracy per image)
        """
        feats = self.detect_features(images)
        return self.scores_from_features(feats)

    @staticmethod
    def msg2str(msg):
        """Convert message to string representation."""
        return "".join([('1' if el else '0') for el in msg])

    @staticmethod
    def str2msg(s):
        """Convert string representation to boolean message."""
        return [True if el == '1' else False for el in s]

    def _transform_image(self, image):
        """Transform PIL image for model input."""
        transform_imnet = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform_imnet(image)
