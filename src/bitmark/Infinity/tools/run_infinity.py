import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import math
import time
import hashlib
import yaml
import argparse
import shutil
import re
import random

import logging
logger = logging.getLogger(__name__)

import cv2
import numpy as np
import torch
torch._dynamo.config.cache_size_limit=64
import pandas as pd
# import detect_watermark
# from detect_watermark import get_detector, WatermarkInference
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast, LogitsProcessor,LogitsProcessorList
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from torch.cuda.amp import autocast
from infinity.models.infinity import Infinity
from infinity.models.basic import *
import PIL.Image as PImage
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
import logging
logger = logging.getLogger(__name__)
from pydantic.utils import deep_update


def extract_key_val(text):
    pattern = r'<(.+?):(.+?)>'
    matches = re.findall(pattern, text)
    key_val = {}
    for match in matches:
        key_val[match[0]] = match[1].lstrip()
    return key_val

def encode_prompt(
  text_tokenizer,
  text_encoder,
  prompts: List[str],
  enable_positive_prompt: bool = False
) -> Tuple[torch.FloatTensor, List[int], torch.IntTensor, int]:
  # positive‐prompt augmentation
  if enable_positive_prompt:
    prompts = [aug_with_positive_prompt(p) for p in prompts]
  # Tokenize the whole batch
  tokens = text_tokenizer(
    text=prompts,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
  )
  input_ids = tokens.input_ids.cuda(non_blocking=True)
  mask   = tokens.attention_mask.cuda(non_blocking=True)
  # Encode
  out = text_encoder(input_ids=input_ids, attention_mask=mask)
  text_features = out['last_hidden_state'].float()  # [B, L, D]
  # Build lengths and cumulative‐sum (cu_seqlens_k)
  lens_tensor = mask.sum(dim=-1)           # torch.Int64Tensor, shape [B]
  lens = lens_tensor.tolist()            # List[int]
  lens_i32 = lens_tensor.to(torch.int32)
  cu_seqlens_k = torch.cat([
    torch.zeros(1, dtype=torch.int32, device=mask.device),
    lens_i32.cumsum(0)
  ], dim=0)                     # shape [B+1]
  cu_seqlens_k = cu_seqlens_k.to(torch.int32)
  # Pack all kv‐pairs into one long tensor
  kv_compact = []
  for L_i, feat_i in zip(lens, text_features.unbind(0)):
    kv_compact.append(feat_i[:L_i])
  kv_compact = torch.cat(kv_compact, dim=0)     # [sum(L_i), D]
  # max length as a Python int
  max_seqlen_k = max(lens)

#   if kv_compact.dim() == 2:
#     kv_compact = kv_compact.unsqueeze(0)

  return kv_compact, lens, cu_seqlens_k, max_seqlen_k

def aug_with_positive_prompt(prompt):
    for key in ['man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person', 'human', 'adult', 'teenager', 'employee', 
                'employer', 'worker', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'son', 'daughter']:
        if key in prompt:
            prompt = prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt

def enhance_image(image):
    for t in range(1):
        contrast_image = image.copy()
        contrast_enhancer = ImageEnhance.Contrast(contrast_image)
        contrast_image = contrast_enhancer.enhance(1.05)  # 增强对比度
        color_image = contrast_image.copy()
        color_enhancer = ImageEnhance.Color(color_image)
        color_image = color_enhancer.enhance(1.05)  # 增强饱和度
    return color_image

def forward(self, prompts):
    logger = logging.getLogger(__name__)
    logger.debug(f"Encoding prompt: {prompts!r}")
    label = encode_prompt(
      self.text_tokenizer,
      self.text_encoder,
      prompts,
      int(self.enable_positive_prompt),
    )
    B = len(prompts)
    with torch.amp.autocast(
      device_type=self.device.type,
      enabled=True,
      dtype=torch.bfloat16,
      cache_enabled=True,
    ):
      _, _, img_list = self.infinity.autoregressive_infer_cfg(
        vae=self.vae,
        scale_schedule=self.scale_schedule,
        label_B_or_BLT=label,
        B=B,
        g_seed=self.seed,
        cfg_list=self.cfg_list,
        tau_list=self.tau_list,
        top_k=0,
        top_p=0.97,
        returns_vemb=1,
        norm_cfg=False,
        cfg_insertion_layer=[self.args.cfg_insertion_layer],
        vae_type=self.args.vae_type,
        ret_img=True,
        trunk_scale=1000,
        sampling_per_bits=self.args.sampling_per_bits,
        inference_mode=True,
      )
    logger.debug("Inference step complete, returning image tensor.")
    return img_list

def gen_one_img(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=0,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    watermark=None,
    scales_injector=None,
    decode_per_scale=False
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None

    label = encode_prompt(
      text_tokenizer,
      text_encoder,
      prompt,
      int(enable_positive_prompt),
    )
    if isinstance(prompt, str):
        B = 1
    else:
        B = len(prompt)


    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        ret, gen_all_bit_indices, img = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=label, g_seed=g_seed,
            B=B, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits, watermark=watermark, scales_injector=scales_injector, decode_per_scale=decode_per_scale
        )
    logger.info(f"cost: {time.time() - sstt}, infinity cost={time.time() - stt}")
    return ret, gen_all_bit_indices, img

def get_prompt_id(prompt):
    md5 = hashlib.md5()
    md5.update(prompt.encode('utf-8'))
    prompt_id = md5.hexdigest()
    return prompt_id

def save_slim_model(infinity_model_path, save_file=None, device='cpu', key='gpt_fsdp'):
    print('[Save slim model]')
    full_ckpt = torch.load(infinity_model_path, map_location=device)
    infinity_slim = full_ckpt['trainer'][key]
    # ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
    if not save_file:
        save_file = osp.splitext(infinity_model_path)[0] + '-slim.pth'
    print(f'Save to {save_file}')
    torch.save(infinity_slim, save_file)
    print('[Save slim model] done')
    return save_file

def load_tokenizer(t5_path =''):
    print(f'[Loading tokenizer and text encoder]')
    # Load model directly

    text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained("google/flan-t5-xl", revision=None, legacy=True)
    text_tokenizer.model_max_length = 512
    text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained("google/flan-t5-xl", torch_dtype=torch.float16)
    text_encoder.to('cuda')
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return text_tokenizer, text_encoder

def load_infinity(
    rope2d_each_sa_layer, 
    rope2d_normalized_by_hw, 
    use_scale_schedule_embedding, 
    pn, 
    use_bit_label, 
    add_lvl_embeding_only_first_block, 
    model_path='', 
    scale_schedule=None, 
    vae=None, 
    device='cuda', 
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    checkpoint_type='torch',
):
    print(f'[Loading Infinity]')
    text_maxlen = 512
    with torch.amp.autocast('cuda',enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: Infinity = Infinity(
            vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
            shared_aln=True, raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=True,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        print(f'[Load Infinity weights]')
        if checkpoint_type == 'torch':
            ckpt = torch.load(model_path, map_location=device)
            # Trainer-format finetune ckpts wrap weights as ckpt['trainer']['gpt_fsdp'];
            # base releases are already a flat state_dict.
            state_dict = ckpt['trainer']['gpt_fsdp'] if isinstance(ckpt, dict) and 'trainer' in ckpt else ckpt
            print(infinity_test.load_state_dict(state_dict))
        elif checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(infinity_test, model_path, strict=False)
        infinity_test.rng = torch.Generator(device=device)
        return infinity_test

def load_visual_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    if args.vae_type in [14,16,18,20,24,32,64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae

def load_transformer(vae, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    if args.checkpoint_type == 'torch': 
        # copy large model to local; save slim to local; and copy slim to nas; load local slim model
        if osp.exists(args.cache_dir):
            local_model_path = osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_'))
        else:
            local_model_path = model_path
        if args.enable_model_cache:
            slim_model_path = model_path.replace('ar-', 'slim-')
            local_slim_model_path = local_model_path.replace('ar-', 'slim-')
            os.makedirs(osp.dirname(local_slim_model_path), exist_ok=True)
            print(f'model_path: {model_path}, slim_model_path: {slim_model_path}')
            print(f'local_model_path: {local_model_path}, local_slim_model_path: {local_slim_model_path}')
            if not osp.exists(local_slim_model_path):
                if osp.exists(slim_model_path):
                    print(f'copy {slim_model_path} to {local_slim_model_path}')
                    shutil.copyfile(slim_model_path, local_slim_model_path)
                else:
                    if not osp.exists(local_model_path):
                        print(f'copy {model_path} to {local_model_path}')
                        shutil.copyfile(model_path, local_model_path)
                    save_slim_model(local_model_path, save_file=local_slim_model_path, device=device)
                    print(f'copy {local_slim_model_path} to {slim_model_path}')
                    if not osp.exists(slim_model_path):
                        shutil.copyfile(local_slim_model_path, slim_model_path)
                        os.remove(local_model_path)
                        os.remove(model_path)
            slim_model_path = local_slim_model_path
        else:
            slim_model_path = model_path
        print(f'load checkpoint from {slim_model_path}')
    elif args.checkpoint_type == 'torch_shard':
        slim_model_path = model_path

    if args.model_type == 'infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    elif args.model_type == 'infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'infinity_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, 
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label, 
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block, 
        model_path=slim_model_path, 
        scale_schedule=None, 
        vae=vae, 
        device=device, 
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        checkpoint_type=args.checkpoint_type,
    )
    return infinity

