# watermarks package
from .post_processing import *
from .in_generation import *

from . import in_generation, post_processing
from .in_generation import bitmark, stablesignature, tree_ring
from .post_processing import rivagan, siren, stegastamp, trustmark

from src.utils.commonargs import compose_generation_config


def build_watermark(config):
    """Construct a watermark from a generation config.

    Reads ``config.watermark.{method|name, type, model_path, params}`` and returns
    ``(watermark_instance, wm_cfg)``.  This is the single extension point for
    adding new watermarks — add one entry here and one config, nothing else.
    """
    config = compose_generation_config(config)

    wm_cfg = config.get("watermark", {})
    wm_type = wm_cfg.get("type")
    wm_name = wm_cfg.get("method")
    wm_model_path = wm_cfg.get("model_path")
    wm_params = wm_cfg.get("params", {})

    if wm_name == "rivagan":
        return RivaGANWatermark(model_path=wm_model_path, params=wm_params, device="cuda"), wm_cfg
    elif wm_name == "stegastamp":
        return StegaStampWatermark(model_path=wm_model_path, params=wm_params, device="cuda"), wm_cfg
    elif wm_name == "trustmark":
        return TrustMarkWatermark(params=wm_params, device="cuda"), wm_cfg
    elif wm_name == "siren":
        return SIRENWatermark(model_path=wm_model_path, params=wm_params, device="cuda"), wm_cfg
    elif wm_name == "treering":
        return TreeRingWatermark(dic=config, device="cuda"), wm_cfg
    elif wm_name == "stablesignature":
        return StableSignatureWatermark(dic=config, device="cuda"), wm_cfg
    elif wm_name == "bitmark":
        return BitMarkWatermark(dic=config, device="cuda"), wm_cfg
    else:
        raise NotImplementedError(f"Unsupported watermark method: {wm_name!r} (type={wm_type!r})")
