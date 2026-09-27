import os
from tools.helper import count_match_after_reencoding,joint_vi_vae_encode_decode, compose_scales_to_image, decode_codes, save_single_image, get_stripped_delta, set_seeds
from detect_watermark import get_detector, get_watermark_scales, detect
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)
import argparse
import torch
from tqdm import tqdm
from tools.run_infinity import add_common_arguments, load_visual_tokenizer 
import json
from pydantic.v1.utils import deep_update
import random
import os.path as osp
def flip(args, patterns:dict|None, detector, img_path:str, img_name:str, save_path:str, watermark_scales:list, vae,factor:float, watermark_image:bool = True) -> dict:
    """Flips bit according to a given pattern for each bit on each scale specified in watermark scales. 
    
    If we want to watermark_image, each last bit in subset s
    in S_R is flipped by probability factor, such that s is in S_G. 

    If we want to remove the watermark_image=False, each last bit in subset s in S_G is flipped by the probability of (n/|s in S_G| -0.5)*factor. 
    n/|s in S_G| hereby is the portion of the green sets within the image. If factor=2, the greensets are removed at the same portion they are currently present within the image

    
    Args:
        args (_type_): args
        patterns (dict): S_R, the patterns we want to change
        dir_path (str): Directory of the images to flip
        img_name (str): Image name of current sample
        save_path (str): Directory to save the flipped images
        watermark_scales (list): On which scales to apply the watermark flipping
        vae (_type_): Encoder/Decoder
        factor (float): By which factor to flip. The severity depends on watermark_image. For watermark_image = False [True], a value of 2.2 [0.08] is recommended. 
        watermark_image (bool, optional): If we want to apply or remove a watermark. Defaults to True.

    Returns:
        dict: Results - z-score, prediction for each scale as well as factor, delta & scales used, as well as the save_path
    """
    results = {"stat_data": {}}
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [ (1, h, w) for (_, h, w) in scale_schedule]
    tgt_h, tgt_w = dynamic_resolution_h_w[args.h_div_w_template][args.pn]["pixel"]
    _, _, main_encoded_bits, _= joint_vi_vae_encode_decode(
        vae, f"{img_path}", scale_schedule, "cuda", tgt_h, tgt_w, apply_spatial_patchify=args.apply_spatial_patchify
    )
    encoded_bits = [main_encoded_bits[i] for i in watermark_scales]  # Only modify watermark on specified scales
    
    ### Determine flipping probabilities (differnet for removing or applying watermark)
    if watermark_image == False: # removes watermark
        flip_probs=[]
        for scale in encoded_bits:
            flattened_scale = torch.cat([t.reshape(-1) for t in scale], dim=0)
            watermark_stats = detector.detect(tokenized_text=flattened_scale)
            flip_prob = (watermark_stats["green_fraction"]-0.5)*factor
            flip_probs.append(flip_prob) # Flip each s in S_R by the probability of ((|S|/|S_G|)-0.5)*factor for the current scale 
        logger.info(flip_probs) 
    else:
        flip_probs = [factor] * len(encoded_bits) # Flip each s in S_R by the probability of factor for every scale in watermark_scales
        assert len(flip_probs) == len(encoded_bits)
    bit_flip = 0
    ### Apply bitflipping according to patterns for each scale separately
    for i, scale in enumerate(encoded_bits):
        current_scale = i + watermark_scales[0]
        if f'scale_{current_scale}' not in results["stat_data"].keys():
            results["stat_data"][f"scale_{current_scale}"] = {'scale':{}}
        bit_flip_scale = 0
        if flip_probs[i]<=0:
            continue
        for j in range(1, scale.shape[-1]):
            mask = torch.zeros_like(scale[...,j], dtype=torch.bool, device="cuda") 
            if patterns: # If we know S_G or S_R
                for pattern in patterns: 
                    mask_a = scale[...,j-1] == int(pattern[0]) 
                    mask_b = scale[...,j] == int(pattern[1])
                    tmp_mask= mask_a & mask_b
                    mask= mask | (tmp_mask & mask_b) # We only flip the last bit, hence we only check if mask_b is true
                    probs = torch.where(mask, torch.tensor(flip_probs[i],device="cuda"), torch.tensor(0.0, device="cuda"))
                    random = torch.rand(probs.shape, device="cuda")
                    sampled_mask = random < probs
                    scale[...,j][sampled_mask] = 1-scale[...,j][sampled_mask]
                bit_flip_scale += torch.sum(sampled_mask).item()
            else: # Random flip
                mask = scale == 1        
                probs = torch.where(mask, torch.tensor(flip_probs[i],device="cuda"), torch.tensor(0.0, device="cuda"))
                random = torch.rand(probs.shape, device="cuda")
                sampled_mask = random < probs
                scale[...,sampled_mask] = 1-scale[...,sampled_mask]
                bit_flip_scale += torch.sum(sampled_mask).item()
        results["stat_data"][f"scale_{current_scale}"]['scale']=  {"bit_flips": bit_flip_scale}
        bit_flip += bit_flip_scale
    results["bit_flips"] = bit_flip
    logger.info(f'flipped bits: {bit_flip}')
    if logger.root.level == logging.INFO:  # Only relevant for initial validation if everything worked
        encoded_bits_flattened = torch.cat([t.reshape(-1) for t in encoded_bits], dim=0)
        metrics = detector.detect(tokenized_text=encoded_bits_flattened)
        if metrics["prediction"] == watermark_image:
            logger.info("succeeded flipping bits")
    
    for i in watermark_scales:
        main_encoded_bits[i]=encoded_bits[i-watermark_scales[0]] # Decoding needs all scales, not just the watermarked ones
    summed_codes = compose_scales_to_image(args, main_encoded_bits, vae, scale_schedule)
    img = decode_codes(summed_codes, vae)
    save_single_image(img, f"{save_path}/{img_name}") 
    ### Detect flipped image
    # Test if the flipped bits actually stay consistent after reencoding, which is what we care for
    metrics = detect(args, f"{save_path}/{img_name}", detector, vae, watermark_scales, True)
    logger.info(metrics['stat_data'])
    results["z_score"] = metrics["z_score"]
    results["prediction"] = metrics['prediction']
    results["stat_data"] = deep_update(results["stat_data"], metrics["stat_data"]) 

    logger.info(results['stat_data'])
    logger.info(f"reecndoing z-value={metrics['z_score']}")
    _, _, reencoded_bits, _= joint_vi_vae_encode_decode(
        vae, f"{save_path}/{img_name}", scale_schedule, "cuda", tgt_h, tgt_w, apply_spatial_patchify=args.apply_spatial_patchify
    )
    ret, num_matches_list, num_total_list = count_match_after_reencoding(
                    main_encoded_bits, reencoded_bits, watermark_scales, compare_only_on_watermarked_scales=False # Maybe set to something else?
                    )
    if metrics["prediction"] == watermark_image:
        logger.info(results)
    results["apply_watermark"] = watermark_image
    results["factor"] = factor
    results["scales"] = watermark_scales
    results["delta"] = get_stripped_delta(args.watermark_delta)
    results["image_path"] = f"{save_path}/{img_name}"
    results["image_id"] = int(img_name.split('.')[0])
    results["stat_data"] = deep_update(results["stat_data"],ret)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--watermark_image", type=int, choices=[-1,0,1])
    parser.add_argument('--flip_factor', type=float)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--gen_metadata_path', type=str)
    args = parser.parse_args()
    set_seeds(args.seed)
    logging.basicConfig(encoding="utf-8", level=logging.WARNING)

    # parse cONfIg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # load vae
    vae = load_visual_tokenizer(args)
    detector = get_detector(args)
    
    watermark_scales = get_watermark_scales(args.watermark_scales, args.h_div_w_template, args.pn)
    if args.watermark_image == 1: # True apply watermark
        pattern = {"00", "11"}
        load_scales = 0
    elif args.watermark_image == 0: # False = remove watermark
        pattern = {"01", "10"}
        load_scales = args.watermark_scales
    else: 
        pattern = None # Flips randomly bits towards 0
        load_scales = args.watermark_scales

    save_path = args.out_dir
    os.makedirs(save_path, exist_ok=True)
    jsonl_list = []
    with open(args.gen_metadata_path, mode="r", encoding="utf-8") as json_file:
        if "coco2014/val2014" in args.gen_metadata_path:
            image_duplicates= []
            cnt = 0
            meta = json.load(json_file)
            annotations = meta["annotations"]
            random.shuffle(annotations)
            for annotation in annotations:
                if cnt== 1000:
                    break
                #assert osp.exists(gt_image_path), gt_image_path
                if annotation["image_id"] in image_duplicates:
                    continue
                image_duplicates.append(annotation["image_id"])
                img_id = annotation["image_id"]
                prompt = annotation["caption"]
                if not prompt:
                    continue
                img_path = osp.join("/path/datasets/coco2014/val2014", f"COCO_val2014_{img_id:012d}.jpg")
                results = flip(args, patterns=pattern, detector=detector, img_path=img_path, img_name=f"{img_id}.png", save_path=save_path, watermark_scales=watermark_scales, vae=vae, factor=args.flip_factor, watermark_image=args.watermark_image)
                results["prompt"] = prompt
                jsonl_list = json.dumps(results, default=str) + "\n"
                jsonl_file = f"{save_path}/metrics.json"
                with open(jsonl_file, "a") as json_file:
                    json_file.writelines(jsonl_list)
                cnt+=1
        else:
            for i, line in enumerate(tqdm(json_file, miniters=50)):        
                stripped = line.strip()
                if not stripped:
                    print(f"Skipping line {i} as strip did not work. Line: {line}, strip: {line.strip()}")
                    skipped_lines+=1
                    continue
                try:
                    line2 = json.loads(line.strip())
                    del line2["stat_data"]
                    if "image_path" in line2.keys():             
                        img_path = line2["image_path"]
                    else:
                        img_path = line2["path"]
                    img_name = str(line2["image_id"])
                    img_name += ".png"
                    prompt = line2['prompt']
                    results = flip(args, patterns=pattern, detector=detector, img_path=img_path, img_name=img_name, save_path=save_path, watermark_scales=watermark_scales, vae=vae, factor=args.flip_factor, watermark_image=args.watermark_image)
                    results["prompt"] = prompt
                    jsonl_list = json.dumps(results, default=str) + "\n"
                    jsonl_file = f"{save_path}/metrics.json"
                    with open(jsonl_file, "a") as f:
                        f.writelines(jsonl_list)
                except Exception as e:
                    print(line2)
                    print(e)        