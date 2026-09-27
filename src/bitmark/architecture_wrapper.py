import os
from pathlib import Path
import torch
import torchvision
from PIL import Image
import numpy as np
import sys
from torchvision.transforms.functional import to_tensor
import argparse
try:
    from .helper import count_match_after_reencoding, get_watermark_scales
except ImportError:
    from helper import count_match_after_reencoding, get_watermark_scales
# Note: Architecture-specific imports are done lazily in each class to avoid environment conflicts

import gc

BITMARK_ROOT = Path(__file__).resolve().parent
INFINITY_ROOT = BITMARK_ROOT / "Infinity"


def _ensure_infinity_path():
    infinity_root = str(INFINITY_ROOT)
    if infinity_root not in sys.path:
        sys.path.insert(0, infinity_root)

class ArchitectureWrapper:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def gen_img(self, prompts):
        # This method should implement the logic to generate an image
        pass


    def shape_img(self, img):
        # This method should implement the logic to shape an image
        pass

class VAEWrapper:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def encode(self, image_or_path):
        if type(image_or_path) == str:
            pil_image = Image.open(image_or_path).convert('RGB')
        elif type(image_or_path) == list:
            pil_image = []
            for img_or_path2_lol in image_or_path:
                if type(img_or_path2_lol) == str: 
                    pil_image.append(Image.open(img_or_path2_lol).convert('RGB'))
                else:
                    pil_image.append(img_or_path2_lol)
        else:
            pil_image = image_or_path
        # This method should implement the logic to encode an image using VAE
        return pil_image

    def decode(self, encoded_img):
        # This method should implement the logic to decode an image using VAE
        pass


    def calc_bit_overlap(self, gen_bit_indices, image_path, batch_idx, watermark_scales=None):
        _,_,encoded_bit_indices, _ = self.encode(image_path)
        gen_bits = gen_bit_indices.reshape(-1)
        encoded_bits = encoded_bit_indices.reshape(-1)
        
        # Calculate bit overlap
        matches = (gen_bits == encoded_bits).sum().item()
        total = gen_bits.numel()
        overlap_ratio = matches / total if total > 0 else 0.0
                
        ret_count = {
            "match_reencoding": matches,
            "total_bits": total,
            "overlap_ratio": overlap_ratio,
        }
        
        return ret_count

class InfinityBAE(VAEWrapper):
    def __init__(self, args):
        super().__init__(args)
        
        # Lazy import for Infinity dependencies
        _ensure_infinity_path()
        from tools.run_infinity import load_visual_tokenizer
        from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
        
        # Store imports as instance variables
        self.dynamic_resolution_h_w = dynamic_resolution_h_w
        self.h_div_w_templates = h_div_w_templates
        
        self.vae = load_visual_tokenizer(args)
        self.apply_spatial_patchify = args.apply_spatial_patchify
        self.scale_schedule, self.vae_scale_schedule, self.tgt_h, self.tgt_w = self.init_scale_schedule(args)       
        self.watermark_scales = get_watermark_scales(args.watermark_scales, self.scale_schedule)

    def encode(self, image_or_path, add_noise):
        pil_image = super().encode(image_or_path)
        inp = self.transform(pil_image, self.tgt_h, self.tgt_w, add_noise)
        img_embedding, z, _, all_bit_indices, _, interpolate_residual_per_scale = self.vae.encode(inp.unsqueeze(0).to(self.device), scale_schedule=self.vae_scale_schedule)
        if False: # patchify operation -- removed, as this breaks BitMark
            for i, idx_Bld in enumerate(all_bit_indices): 
                idx_Bld = idx_Bld.squeeze(1)
                idx_Bld = idx_Bld.permute(0, 3, 1, 2)                       # [B, d, h, w] (from [B, h, w, d])
                idx_Bld = torch.nn.functional.pixel_unshuffle(idx_Bld, 2)    # [B, 4d, h//2, w//2]
                idx_Bld = idx_Bld.permute(0, 2, 3, 1)                       # [B, h//2, w//2, 4d]
                all_bit_indices[i] = idx_Bld.unsqueeze(1) # [B, 4d, h, w]
        #recons_img = vae.decode(z)[0]
        #logger.info(f'recons: z.shape: {z.shape}, recons_img shape: {recons_img.shape}')
        #t3 = time.time()
        #logger.info(f'vae encode takes {t2-t1:.2f}s, decode takes {t3-t2:.2f}s')
        #recons_img = (recons_img + 1) / 2
        #recons_img = recons_img.permute(0, 2, 3, 1).mul_(255).cpu().numpy().astype(np.uint8)
        #gt_img = (inp[0] + 1) / 2
        #gt_img = gt_img.permute(0, 2, 3, 1).mul_(255).cpu().numpy().astype(np.uint8)
        return _, interpolate_residual_per_scale, all_bit_indices, img_embedding
    
    def init_scale_schedule(self, args): 
        h_div_w_template = self.h_div_w_templates[
            np.argmin(np.abs(self.h_div_w_templates - 1)) # NOTE insert proper value
        ]
        scale_schedule = self.dynamic_resolution_h_w[h_div_w_template][args.pn]["scales"]
        scale_schedule = [(1, h, w) for (t, h, w) in scale_schedule]

        if args.apply_spatial_patchify:
            vae_scale_schedule = [
                (pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule
            ]
        else:
            vae_scale_schedule = scale_schedule
        tgt_h, tgt_w = self.dynamic_resolution_h_w[h_div_w_template][args.pn]["pixel"]
        return scale_schedule, vae_scale_schedule, tgt_h, tgt_w

    def transform(self, pil_img, tgt_h, tgt_w, add_noise):
        #tmp_list = []
        #for pil_img in pil_imgs:
        width, height = pil_img.size
        if width / height <= tgt_w / tgt_h:
            resized_width = tgt_w
            resized_height = int(tgt_w / (width / height))
        else:
            resized_height = tgt_h
            resized_width = int((width / height) * tgt_h)
        pil_img = pil_img.resize((resized_width, resized_height), resample=Image.LANCZOS)
        # crop the center out
        arr = np.array(pil_img)
        crop_y = (arr.shape[0] - tgt_h) // 2
        crop_x = (arr.shape[1] - tgt_w) // 2
        im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
        # print("Tensor shape before noise:", im.shape)
        im = im + (0.003)*torch.randn(im.shape, device=im.device)  # Add noise 
        # print("Noise Enabled")
        im = im.add(im).add_(-1)
        return im

    def calc_bit_overlap(self, gen_bit_indices, image_path_or_bits, batch_idx, watermark_scales = None):
        if type(image_path_or_bits) == str:
            gt_img, interpolated_residual_per_scale, encoding_bit_indices, _ = self.encode(image_path, False)
        else:
            encoding_bit_indices = image_path_or_bits
        current_gen_bit_indices = [indices[batch_idx,::] for indices in gen_bit_indices]
        ret_count, num_matches_list, num_total_list = count_match_after_reencoding(
            encoding_bit_indices, current_gen_bit_indices, watermark_scales, compare_only_on_watermarked_scales=False # Maybe set to something else?
        )

        matches = sum(num_matches_list)
        total = sum(num_total_list)
        overlap_ratio = matches / total if total > 0 else 0.0
        ret_count = {
            "match_reencoding": matches,
            "total_bits": total,
            "overlap_ratio": overlap_ratio,
        }


        return ret_count

class Infinity(ArchitectureWrapper):
    def __init__(self, args, vae_wrapper):
        # load text encoder
        
        args.cfg = list(map(float, args.cfg.split(",")))
        if len(args.cfg) == 1:
            args.cfg = args.cfg[0]
        self.args = args
        
        # Lazy import for Infinity dependencies
        _ensure_infinity_path()
        from tools.run_infinity import load_transformer, load_tokenizer, gen_one_img
        
        # Store the imported function for later use
        self.gen_one_img = gen_one_img
        
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load infinity
        self.infinity = load_transformer(vae_wrapper.vae, args)

        self.scale_schedule = vae_wrapper.scale_schedule
        self.vae_scale_schedule = vae_wrapper.vae_scale_schedule
        self.tgt_h = vae_wrapper.tgt_h
        self.tgt_w = vae_wrapper.tgt_w
        
        self.scales_injector = None
        #scales_injector = ScalesInjector(args, vae, scale_schedule, tgt_h, tgt_w)
        

    def gen_img(self, prompts, vae, watermark_inference):
        _, _, img = self.gen_one_img(
            self.infinity,
            vae,
            self.text_tokenizer,
            self.text_encoder,
            prompt=prompts,
            g_seed=self.args.seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=self.args.cfg,
            tau_list=self.args.tau,
            scale_schedule=self.scale_schedule,
            cfg_insertion_layer=[self.args.cfg_insertion_layer],
            vae_type=self.args.vae_type,
            sampling_per_bits=self.args.sampling_per_bits,
            enable_positive_prompt=self.args.enable_positive_prompt,
            watermark=watermark_inference,
            scales_injector=self.scales_injector,
            decode_per_scale=self.args.decode_per_scale,
        )
        return img
    
def get_architecture(args, vae_wrapper=None):
    """
    Get the architecture based on the provided arguments.
    """
    if args.architecture == "infinity":
        # Load Infinity config file
        return Infinity(args, vae_wrapper)
    elif args.architecture == "big_r":
        return BiGR(args)
    elif args.architecture == "instella_iar":
        return InstellaIAR(args)
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")
    
def get_vae(args):
    """
    Get the VAE based on the provided arguments.
    """
    if args.architecture == "infinity":
        return InfinityBAE(args)
    elif args.architecture == "big_r":
        return BiGRBAE(args)
    elif args.architecture == "instella_iar":
        return InstellaBAE(args)
    else:
        raise ValueError(f"Unsupported architecture for VAE: {args.architecture}")
    
def get_architecture_arguments():
    # First pass: parse only the architecture argument to determine which architecture to use
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--architecture", type=str, required=True, 
                           help="Architecture to use (e.g., 'infinity', 'instella_iar', etc.)")
    pre_args, remaining_args = pre_parser.parse_known_args()
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    match (pre_args.architecture):
        case "infinity":
            add_infinity_arguments(parser)
        case "big_r":
            add_big_r_arguments(parser)
        case "instella_iar":
            add_instella_iar_arguments(parser)
        case _:
            raise ValueError(f"Unsupported architecture: {pre_args.architecture}")
    # Second pass: create full parser with architecture-specific arguments

    return parser

def add_common_arguments(parser):
    parser.add_argument("--architecture", type=str, required=True, 
                           help="Architecture to use (e.g., 'infinity', 'instella_iar', etc.)")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--watermark_scales", type=int, default=0)
    parser.add_argument("--watermark_context_width", type=int, default=4)
    parser.add_argument("--watermark_seeding_scheme", type=str, default="selfhash")
    parser.add_argument("--watermark_delta", type=float, default=1.0)
    parser.add_argument("--watermark_gen_image", type=int, default=1, choices=[0,1])
    parser.add_argument("--watermark_count_bit_loss_after_reencoding", type=int, default=0, choices=[0,1])
    parser.add_argument("--watermark_method", type=str, default='2-bit_pattern')
    parser.add_argument("--watermark_count_bit_flip", type=int, default=0, choices=[0,1])
    parser.add_argument("--watermark_add_noise", type=int, default=0, choices=[0,1])
    parser.add_argument("--watermark_remove_duplicates", type=int, default=0, choices=[0,1])
    parser.add_argument("--set", type=str)
    parser.add_argument('--seed', type=int, default=0)

def add_infinity_arguments(parser):
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=1)
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--enable_model_cache', type=int, default=0, choices=[0,1])
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])
    parser.add_argument("--inject_scales", type=int, default = 0, choices=[0,1,2])
    parser.add_argument("--inject_scales_path", type=str, default = '')
    parser.add_argument("--decode_per_scale", type=int, default=0, choices=[0,1])
