import copy
import os

import numpy as np
import scipy
import torch
from diffusers import DPMSolverMultistepScheduler

from src.model_backends.ldm.sd_21_base.inverse_stable_diffusion import InversableStableDiffusionPipeline
from src.utils.optim_utils import get_dataset, transform_img

from ..base import GenerationWatermark


def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


def get_watermarking_mask(init_latents_w, args, device):
    args = args.watermark.params
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == "circle":
        np_mask = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)
        if args.w_channel == -1:
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == "square":
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            watermarking_mask[
                :,
                :,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
            ] = True
        else:
            watermarking_mask[
                :,
                args.w_channel,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
                anchor_p - args.w_radius : anchor_p + args.w_radius,
            ] = True
    elif args.w_mask_shape == "no":
        pass
    else:
        raise NotImplementedError(f"w_mask_shape: {args.w_mask_shape}")

    return watermarking_mask


def get_watermarking_pattern(pipe, args, device, shape=None):
    args = args.watermark.params
    # Local generator so building gt_patch does not perturb the global RNG —
    # otherwise the generation loop's noise stream would depend on whether
    # gt_patch.pt was cached on disk or freshly built this run.
    gt_generator = torch.Generator(device=device).manual_seed(int(args.w_seed))
    if shape is not None:
        gt_init = torch.randn(*shape, device=device, generator=gt_generator)
    else:
        gt_init = pipe.get_random_latents(generator=gt_generator)

    gt_init = gt_init.float()

    if "seed_ring" in args.w_pattern:
        gt_patch = gt_init
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif "seed_zeros" in args.w_pattern:
        gt_patch = gt_init * 0
    elif "seed_rand" in args.w_pattern:
        gt_patch = gt_init
    elif "rand" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif "zeros" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif "const" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif "ring" in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    else:
        raise NotImplementedError(f"w_pattern: {args.w_pattern}")
    return gt_patch


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args):
    device = init_latents_w.device
    watermarking_mask = watermarking_mask.to(device)
    gt_patch = gt_patch.to(device)
    args = args.watermark.params
    init_latents_w = init_latents_w.float()

    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if args.w_injection == "complex":
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == "seed":
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        raise NotImplementedError(f"w_injection: {args.w_injection}")

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real
    init_latents_w = init_latents_w.half()
    return init_latents_w


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    args = args.watermark.params
    if "complex" in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif "seed" in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        raise NotImplementedError(f"w_measurement: {args.w_measurement}")

    if "l1" in args.w_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        raise NotImplementedError(f"w_measurement: {args.w_measurement}")

    return no_w_metric, w_metric


def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])

    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w


class TreeRingWatermark(GenerationWatermark):
    def __init__(self, device = None, dic=None):
        super().__init__(device, dic)
        model_path = self.resolve_m1_pipeline_path()
        if not model_path:
            raise ValueError("model.model_path (or model.model_id) must be set for TreeRing")
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler", cache_dir=model_path
        )
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
            model_path, scheduler=self.scheduler, torch_dtype=torch.float16, revision="fp16", cache_dir=model_path
        )
        self.pipe = self.pipe.to(device)
        # Disable diffusers' internal tqdm progress bars for inference steps
        self.pipe.set_progress_bar_config(disable=True, leave=False)
        
                
        if hasattr(self.dic, "datasets") and hasattr(self.dic.datasets, "output_dir"):
            os.makedirs(self.dic.datasets.output_dir, exist_ok=True)


        print(self.dic)

        if self.mode == "clean_generate":
            self.gt_patch = None
            self.watermarking_mask = None
        elif not os.path.exists(os.path.join(self.dic.datasets.output_dir, "watermarking_mask.pt")):
            print(f"Storing watermarking information at {self.dic.datasets.output_dir}")
            # Shape probe only: get_watermarking_mask reads init_latents_w.shape.
            # Sampling here (as the previous code did via get_random_latents) would
            # advance the global RNG before the generation loop, desyncing
            # watermarked latents from the clean run on first build (cache-miss).
            shape = (
                1,
                self.pipe.unet.config.in_channels,
                self.pipe.unet.config.sample_size,
                self.pipe.unet.config.sample_size,
            )
            init_latents_w = torch.zeros(shape, device=self.device)
            self.gt_patch = get_watermarking_pattern(self.pipe, self.dic, self.device)
            self.watermarking_mask = get_watermarking_mask(init_latents_w, self.dic, self.device)
            torch.save(self.watermarking_mask, os.path.join(self.dic.datasets.output_dir, "watermarking_mask.pt"))
            torch.save(self.gt_patch, os.path.join(self.dic.datasets.output_dir, "gt_patch.pt"))
        else:
            print(f"Loading presaved watermarking information from {self.dic.datasets.output_dir}")
            self.gt_patch = torch.load(os.path.join(self.dic.datasets.output_dir, "gt_patch.pt")).to(self.device)
            self.watermarking_mask = torch.load(os.path.join(self.dic.datasets.output_dir, "watermarking_mask.pt")).to(self.device)
            
    def _generate_watermarked(self, prompts):
        dataset_params = self.dic.dataset_params
        assert dataset_params.num_images_per_prompt == 1, "Its using batched generation, can only generate 1 image per prompt."
        batch_size = len(prompts)
        # Fresh latents per image; same gt_patch/mask fingerprint shared across the batch.
        init_latents = torch.cat(
            [self.pipe.get_random_latents().to(self.device) for _ in range(batch_size)], dim=0
        )
        mask_b = self.watermarking_mask.expand(batch_size, -1, -1, -1)
        patch_b = self.gt_patch.expand(batch_size, -1, -1, -1)
        init_latents_w = inject_watermark(init_latents, mask_b, patch_b, self.dic)
        outputs = self.pipe(
            prompts,
            num_images_per_prompt=1,
            guidance_scale=dataset_params.guidance_scale,
            num_inference_steps=dataset_params.num_inference_steps,
            height=dataset_params.image_length,
            width=dataset_params.image_length,
            latents=init_latents_w,
        )
        return list(outputs.images)

    def _generate_clean(self, prompts):
        dataset_params = self.dic.dataset_params
        assert dataset_params.num_images_per_prompt == 1, "Its using batched generation, can only generate 1 image per prompt."
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

    def detect_features(self, images):
        dataset_params = self.dic.dataset_params

        tester_prompt = ""  # assume at the detection time, the original prompt is unknown
        text_embeddings = self.pipe.get_text_embedding(tester_prompt)

        out = []

        for i in images:
            img = transform_img(i).unsqueeze(0).to(text_embeddings.dtype).to(self.device)

            image_latents_no_w = self.pipe.get_image_latents(img, sample=False)

            reversed_latents_no_w = self.pipe.forward_diffusion(
                latents=image_latents_no_w,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=dataset_params.num_inference_steps,
            )
            # print(f"Shape: {reversed_latents_no_w.shape}, {reversed_latents_no_w}")
            out.append(reversed_latents_no_w)

        return out

    def scores_from_features(self, features):
        scores = []
        for f in features:
            no_w_metric, _ = eval_watermark(
                f, f, self.watermarking_mask, self.gt_patch, self.dic
            )
            scores.append(-no_w_metric)

        return scores

    def detect(self, images):
        """
        images: List[PIL.Image]
        returns: List[float]
        """
        dataset_params = self.dic.dataset_params

        tester_prompt = ""  # assume at the detection time, the original prompt is unknown
        text_embeddings = self.pipe.get_text_embedding(tester_prompt)

        scores = []

        for i in images:
            img = transform_img(i).unsqueeze(0).to(text_embeddings.dtype).to(self.device)

            image_latents_no_w = self.pipe.get_image_latents(img, sample=False)

            reversed_latents_no_w = self.pipe.forward_diffusion(
                latents=image_latents_no_w,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=dataset_params.num_inference_steps,
            )

            no_w_metric, _ = eval_watermark(
                reversed_latents_no_w, reversed_latents_no_w, self.watermarking_mask, self.gt_patch, self.dic
            )
            scores.append(-no_w_metric)

        return scores


def main(args) -> None:
    # dataset
    dataset, prompt_key = get_dataset(args)
