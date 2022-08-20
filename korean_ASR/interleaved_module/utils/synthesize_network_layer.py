import logging

import torch

from typing import Union
from typing import Any
from pathlib import Path

from korean_ASR.interleaved_module.utils.load_module import forge_stats_file
from korean_ASR.interleaved_module.utils.load_module import build_asr_model, load_param


def model_synthesize(
        logger: logging.RootLogger,
        conformer_model_path: Union[Path, str] = None,
        language_model_path: Union[Path, str] = None,
        interleaved_config_path: Union[Path, str] = None,
        save_path: Union[Path, str] = None
) -> Any:
    """

    """
    logger.info("")
    logger.info(f"Importing Interleaved conformer module...")
    logger.info(f"config file from {interleaved_config_path}...")
    forge_stats_file(
        config_path=interleaved_config_path,
        save_path=save_path,
    )
    interleaved_conformer = build_asr_model(interleaved_config_path)
	
    logger.info("")
    logger.info(f"loading param_dict from pretrained conformer & pretrained language model...")
    logger.info(f"pretrained conformer path: {conformer_model_path}")
    logger.info(f"pretrained language model path: {language_model_path}")
    conformer_param = load_param(conformer_model_path)
    lm_param = load_param(language_model_path)
    interleaved_param = interleaved_conformer.state_dict()
	
    logger.info("")
    logger.info(f"Fusing conformer encoder and LM decoder...")
    logger.info(f"Constructing pretrained interleaved model...")
    new_lm_state_dict = {}

    # Remove lm prefix for LM param_dict keys
    for k in lm_param:
        if k.startswith("lm"):
            new_k = k.replace("lm.", "")
            new_lm_state_dict[new_k] = lm_param[k]

    lm_param = new_lm_state_dict
    new_lm_state_dict = {}

    # Switch LM transformer encoder param dict keys
    # To Interleaved transformer decoder param dict keys
    for k in lm_param:
        if k.startswith("decoder."):
            new_k = k.replace("decoder.", "decoder.output_layer.")
            new_lm_state_dict[new_k] = lm_param[k]
            continue
        elif k.startswith("encoder.encoders."):
            new_k = k.replace("encoder.encoders.", "decoder.decoders.")
            new_lm_state_dict[new_k] = lm_param[k]
            continue
        elif k.startswith("encoder."):
            new_k = k.replace("encoder.", "decoder.")
            new_lm_state_dict[new_k] = lm_param[k]
            continue
        else:
            new_lm_state_dict[k] = lm_param[k]

    # Switch 2 transformer encoder layer to one interleaved decoder layer
    lm_param = new_lm_state_dict
    new_lm_state_dict = {}
    for k in lm_param:
        if k.startswith("decoder.decoders."):
            order = int(k[17])
            if order % 2 == 0:
                new_k = k[:17] + str(order // 2) + k[18:]
                new_k = new_k.replace("feed_forward", "self_feed_forward")
            else:
                new_k = k[:17] + str(order // 2) + k[18:]
                new_k = new_k.replace("self_attn", "src_attn")
                new_k = new_k.replace("feed_forward", "src_feed_forward")
                new_k = new_k.replace("norm1", "norm3")
                new_k = new_k.replace("norm2", "norm4")
        else:
            new_k = k
        new_lm_state_dict[new_k] = lm_param[k]

    lm_param = new_lm_state_dict
    # fuse to construct interleaved model parameter
    for k in interleaved_param:
        if k.startswith("decoder"):
            interleaved_param[k] = lm_param[k]
        else:
            interleaved_param[k] = conformer_param[k]
	
    logger.info("")
    logger.info(f" ===== saving interleaved model: =====")
    logger.info(f" ===== {save_path}/interleaved_conformer.pth =====")
    torch.save(interleaved_param, f"{save_path}/interleaved_conformer.pth")

