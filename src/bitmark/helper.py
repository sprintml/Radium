import cv2
import logging
logger = logging.getLogger(__name__)
import os.path as osp
import numpy as np
from PIL import Image
import PIL.Image as PImage
from torchvision.transforms.functional import to_tensor
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import random
def save_images(img_list, save_path):
    for idx, img in enumerate(img_list):
        tmp_path = save_path[:-4] + str(idx) + save_path[-4:]
        cv2.imwrite(tmp_path, img.cpu().numpy())
    logger.info(f".. saved last image to {osp.abspath(tmp_path)}")

def save_single_image(img_maybe_batch, save_path):
    if type(save_path) != list:
        save_path = [save_path]
    if len(img_maybe_batch.shape) == 3:
        num = 1
    else:
        num = img_maybe_batch.shape[0]
    for i in range(num):
        if num != img_maybe_batch.shape[0]:
            img = img_maybe_batch
        else:
            img = img_maybe_batch[i,::]
        
        if hasattr(img, 'detach'):  # It's a tensor
            img_np = img.detach().cpu().numpy()
        else:
            img_np = img
        
        # Ensure correct data type and range for PIL
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            # Assume values are in [0, 1] range, convert to [0, 255]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                # Assume values are already in [0, 255] range
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        image = PImage.fromarray(img_np, mode='RGB')
        try:
            image.save(save_path[i])
            logger.info(f"Save to {osp.abspath(save_path[i])}")
        except Exception as e:
            logger.warning(f"Failed saving image at {osp.abspath(save_path[i])}: {e}")
            
def get_stripped_delta(delta):
    stripped_delta = ''.join(str(delta).split('.')) # only needed for save path
    if stripped_delta[-1] == "0": # for example join(split(delta)), delta = 20.0 -> 200 -> 20 But for join(split(0.4))-> 04
        stripped_delta = stripped_delta[:-1] 
    return stripped_delta

def decode_codes(summed_codes, vae, output_format='bgr'):
    """
    Decode codes to image tensor.
    
    Args:
        summed_codes: Encoded image codes
        vae: VAE model
        output_format: 'bgr' for cv2 saving, 'rgb' for PIL/matplotlib display
    
    Returns:
        Image tensor in specified format
    """
    img = vae.decode(summed_codes.squeeze(-3))
    img = (img + 1) / 2
    img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8)
    
    return img.squeeze(0)        
        
def compose_scales_to_image(args, bit_encoding, vae_wrapper, scale_schedule, save_scale_img_path=""):
    summed_codes=0
    if args.apply_spatial_patchify:
        scale_schedule = [
            (pt, ph*2, pw*2) for pt, ph, pw in scale_schedule
        ]
    for idx, scale_indices in enumerate(bit_encoding):

        pn = scale_schedule[idx]
        #idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
        idx_Bld = scale_indices.reshape(1, pn[1], pn[2], -1) 
        if False: # unpatchify operation
            idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
            idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
            idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
        idx_Bld = idx_Bld.unsqueeze(1)
        codes = vae_wrapper.vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
        #logger.info(codes.shape)
        codes = F.interpolate(codes, size=scale_schedule[-1], mode=vae_wrapper.vae.quantizer.z_interplote_up)
        #logger.info(codes.shape)
        summed_codes += codes
        if False:
            img = decode_codes(summed_codes.squeeze(-3), vae_wrapper.vae)
            save_single_image(img, f"remove_noise{idx}.png")
        
            img = decode_codes(codes.squeeze(-3), vae_wrapper.vae)
            save_single_image(img, f"remove_noise_code{idx}.png")
    img = decode_codes(summed_codes.squeeze(-3), vae_wrapper.vae)
    if save_scale_img_path != "":
        save_single_image(img, save_scale_img_path)

    return summed_codes.squeeze(-3)

def count_match_after_reencoding(
    encoding_bit_indices, gen_bit_indices: list[torch.Tensor], watermark_scales: list[int], interpolated_residual_per_scale = None, count_bit_match:bool = True, compare_only_on_watermarked_scales:bool = True
) -> tuple[list[int], list[int]]:
    """Count how much bit or token information is lost after reencoding

    Args:
        encoding_bit_indices (list[torch.Tensor]): Quantized residuals (bits) after encoding
        gen_bit_indices (list[torch.Tensor]): Quantized residuals (bits) after generation before decoding
        watermark_scales (list[int]): Which scales have been watermarked
        count_bit_match (bool): Specify if matches should be checked per token or per bit 
        compare_only_on_watermarked_scales (bool): If the matches should be counted only on scales that have been watermarked
    Returns:
        Tuple[list[int], list[int]]: Number of matches for each scale and of total bits or tokens.
    """
    # if compare_only_on_watermarked_scales:
    #     encoding_bit_indices = [encoding_bit_indices[i] for i in watermark_scales] 
    #     gen_bit_indices = [gen_bit_indices[i] for i in watermark_scales]
    num_matches_list = []
    num_total_list = []
    ret = {}
    token_size = gen_bit_indices[0].shape[-1]
    for i, (ebi, gbi) in enumerate(zip(encoding_bit_indices, gen_bit_indices)):
        function = lambda d, idx :torch.sum(torch.eq(d[0][idx],d[1][idx]).to(torch.int32)).to(torch.int32).cpu().item()
        indices_shape = ebi.shape
        scales, token, bit_of_tokens = analyse_bits_of_different_levels(data=(ebi.reshape(-1), gbi.reshape(-1)), function = function, token_size=token_size, seq_len=ebi.reshape(-1).shape[-1])
        ret[f"scale_{i}"] = {"scale": {"match_reencoding": scales}, "token": {"match_reencoding": token}, "bit_n_of_tokens": {"match_reencoding":bit_of_tokens} }
        #print(interpolated_residual_per_scale)
        if interpolated_residual_per_scale != None:
            absolute_interpolated_residual = torch.abs(interpolated_residual_per_scale[i])
            mask = ebi==gbi
            mean_matching = absolute_interpolated_residual[mask].float().mean()
            mean_non_matching = absolute_interpolated_residual[~mask].float().mean()
            logger.info("********")
            logger.info(f"Scale {i}")
            logger.info("\n Absolute MEAN")
            logger.info("Mean of matching absolute values:", round(mean_matching.item(),3))
            logger.info("Mean of non-matching absolute values:", round(mean_non_matching.item(),3))

            std_matching = absolute_interpolated_residual[mask].float().std()
            std_non_matching = absolute_interpolated_residual[~mask].float().std()
            logger.info("\n STANDARD DEVIATION")
            logger.info("Std of matching values:", round(std_matching.item(),3))
            logger.info("Std of non-matching values:", round(std_non_matching.item(),3))
            
            mean_matching = interpolated_residual_per_scale[i][mask].float().mean()
            mean_non_matching = interpolated_residual_per_scale[i][~mask].float().mean()
            
            logger.info("\n MEAN")
            logger.info("Mean of matching values:", round(mean_matching.item(),3))
            logger.info("Mean of non-matching values:", round(mean_non_matching.item(),3))
        
        if count_bit_match: # counts matches per bit
            num_total = indices_shape[2] * indices_shape[3] * indices_shape[4] # num of bits in this scale
            num_matches = scales
        else: # counts matches per token
            num_total = indices_shape[2] * indices_shape[3] # num of tokens in this scale
            matches = torch.where(matches == 32, 1, 0)
            num_matches = matches.reshape(-1).sum(0).item()
        num_total_list.append(num_total)
        num_matches_list.append(num_matches)
    num_total_list = np.array(num_total_list)
    num_matches_list = np.array(num_matches_list)
    for scale in range(len(num_total_list)):
        logger.info(f"{num_matches_list[scale]}/{num_total_list[scale]}, {num_matches_list[scale]/num_total_list[scale]*100:.2f}%")
    return ret, num_matches_list, num_total_list

def analyse_bits_of_different_levels(data, function, token_size, seq_len):
    # Does not work on batches
    num_tokens = int(seq_len/token_size)
    bit_of_tokens = []
    token = []
    #scale = function(data, slice(None))
    scale = function(data, slice(None))
    #for n in range(num_tokens):
    #    token.append(function(data, slice(n*token_size,n*token_size+token_size-1)))
    #for n in range(token_size):
    #    bit_of_tokens.append(function(data, slice(n, seq_len, token_size))) # start, stop, step

    return scale, token, bit_of_tokens

def count_consecutive_sequences(array):
    if len(array) == 0:
        return {}

    # Find the indices where the value changes
    change_indices = np.where(np.diff(array) != 0)[0] + 1
    # Include the start and end indices to capture all sequences
    indices = np.concatenate(([0], change_indices, [len(array)]))

    # Store sequences and their counts
    sequences = []
    
    for start, end in zip(indices[:-1], indices[1:]):
        sequence = end-start
        # sequence = tuple(array[start:end])  # Use tuple for hashable type
        sequences.append(int(sequence))

    # Count frequency of each sequence
    return sequences

def rotate_features(encoded_bits, angle = 90):
    for i, scale in enumerate(encoded_bits):
        scale = scale.squeeze(0).squeeze(0)
        feature_map = scale.permute(2,0,1).unsqueeze(0).float()   # (1, D, H, W) B feature tensor
        rotated_feature_map = torch.stack([
            rotate(feature_map[0, d].unsqueeze(0), angle, interpolation=InterpolationMode.BILINEAR).squeeze(0)
        for d in range(feature_map.shape[1])]).unsqueeze(0) 
        encoded_bits[i] = torch.round(rotated_feature_map.permute(0, 2, 3, 1))  # (1, N, N, D)
    return encoded_bits

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def count_matching_n_bit_sequences(a, b, max_n=32):
    if type(a) == list:
        a = torch.cat([t.reshape(-1) for t in a], dim=0)
        b = torch.cat([t.reshape(-1) for t in b], dim=0)
    res = []
    for n in range(1, max_n + 1):
        # For n, create rolling windows: shape will be (num_windows, n)
        a_wins = a.unfold(0, n, 1)  # shape: (length-n+1, n)
        b_wins = b.unfold(0, n, 1)
        # Compare elementwise, then .all(dim=1) for each window
        matches = (a_wins == b_wins).all(dim=1)
        count = matches.sum().item()
        res.append(count)
    return res

def get_watermark_scales(watermark_scales_config, scale_schedule):
        match watermark_scales_config: 
            case 7: watermark_scales = range(len(scale_schedule))[:12]
            case 6: watermark_scales = range(len(scale_schedule))[:11]
            case 5: watermark_scales = range(len(scale_schedule))[5:]
            case 4: watermark_scales = range(len(scale_schedule))[10:]
            case 3: watermark_scales = range(len(scale_schedule))[:10]
            case 2: watermark_scales = range(len(scale_schedule)) # all scales
            case 1: watermark_scales = [len(scale_schedule)-1] # last scale
            case 0: watermark_scales = [] # No watermarking
            case _: raise(NotImplementedError)
        return watermark_scales