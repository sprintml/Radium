import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import logging
logger = logging.getLogger(__name__)
torch._dynamo.config.cache_size_limit = 64
import logging
from src.bitmark.extended_watermark_processor import WatermarkDetector, WatermarkLookupProcessor
from src.bitmark.helper import get_watermark_scales
from pydantic.utils import deep_update
import time
def detect(
    args, 
    img_path: str,
    watermark_detector: WatermarkDetector,
    vae_wrapper,
    watermark_scales: list= [],
    detect_on_each_scale: bool = False,
    encoding_bit_indices = None,
)-> dict:
    """_summary_

    Args:
        args: Passing all file arguments. I was lazy
        img_path (str): Path to image to detect the watermark on
        watermark_detector (WatermarkDetector): Watermark detector
        vae (_type_, optional): VAE Encoder and quantizer. Defaults to None.
        watermark_scales (list, optional): On which scales the watermark has been applied. Defaults to [].

    Returns:
        dict: Metrics regarding dectection (green tokens count, z-value..)
    """
    metrics = {"stat_data": {}}
    start = time.time()
    if encoding_bit_indices is None:
        _, _, encoding_bit_indices, _ = vae_wrapper.encode(img_path,args.watermark_add_noise)
    if args.architecture == "infinity":    
        if detect_on_each_scale:
            encoding_bit_indices_scale = [encoding_bit_indices[i] for i in range(len(vae_wrapper.scale_schedule))]

        if vae_wrapper.watermark_scales: # Detect only on watermarked scales
            encoding_bit_indices = [encoding_bit_indices[i] for i in vae_wrapper.watermark_scales]
        encoding_bit_indices = torch.cat([t.view(-1, t.shape[-1]) for t in encoding_bit_indices], dim=0)
    if args.watermark_remove_duplicates:
        encoding_bit_indices = torch.unique(encoding_bit_indices.view(-1, encoding_bit_indices.shape[-1]), dim=0)


    encoding_bit_indices_flattened = encoding_bit_indices.flatten() # Reshape to [batch_size, 32]
    watermark_metrics = watermark_detector.detect(tokenized_text=encoding_bit_indices_flattened)
    metrics["time"] = time.time()-start
    if watermark_scales:
        start_scale = watermark_scales[0]
    else:
        start_scale = 0
    
    if detect_on_each_scale: # Detects separately, increases inference time a lot
        for i, scale in enumerate(encoding_bit_indices_scale): 
            flattened_scale = torch.cat([t.reshape(-1) for t in scale], dim=0)
            result = watermark_detector.detect(tokenized_text=flattened_scale)
            metrics["stat_data"][f"scale_{i+start_scale}"] = {'scale': {}}
            metrics["stat_data"][f"scale_{i+start_scale}"]['scale'] = {"z_score":round(result["z_score"], 5), "green_fraction": round(result["green_fraction"],5)}
        watermark_metrics = deep_update(watermark_metrics, metrics)


    return watermark_metrics

class WatermarkInference():
    def __init__(self,args, vae_wrapper = None):
        self.delta = args.watermark_delta
        self.method = args.watermark_method
        
        # print(args.set)
        self.count_bit_flip=args.watermark_count_bit_flip
        self.green_list= set(args.set.split(','))
        self.context_width = len(list(self.green_list)[0])
        if args.architecture == "infinity":

            self.scales = get_watermark_scales(args.watermark_scales, vae_wrapper.scale_schedule)
            if self.scales == []:
                assert self.delta == 0, 'Delta must be 0 if no watermark is applied.'

        match(self.method):
            case '2-bit_pattern':
                self.logits_processor = WatermarkLookupProcessor(vocab=[0,1], device = "cuda", delta=self.delta, green_list=self.green_list) 
            case _:
                raise NotImplementedError
        
def get_detector(args):
    watermark_detector = WatermarkDetector(
        vocab=[0,1],
        gamma=0.5,
        delta=args.watermark_delta,
        device="cuda",
        z_threshold=4.0,
        ignore_repeated_ngrams=False,
        green_list=args.set
    )

    return watermark_detector