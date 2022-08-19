import os
import sys
import re
import torch
import yaml
import argparse
from typing import Union
from typing import Any
from pathlib import Path

import numpy as np

from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.lm import LMTask


def forge_stats_file(
        config_path: Union[Path, str] = None,
        save_path: Union[Path, str] = None,
) -> Any:
    """The function is to counterfeit the Stats file.
       Stats file is needed for the configure of Global Mean Variance Module.
       The Stats file consists of [counts, sum, sum_squared] used for calculating variance."""
    save_path = Path(save_path) / "dummy.npz"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    # numpy array size (n_mel,)
    n_mel = 80
    count = np.ones((n_mel))
    sum = np.ones((n_mel))
    sum_square = np.ones((n_mel))
    np.savez(save_path, count=count, sum=sum, sum_square=sum_square)

    args['normalize_conf']['stats_file'] = str(save_path)

    with config_path.open('w', encoding="utf-8") as f:
        yaml.dump(args, f, default_flow_style=False)

def build_asr_model(
        config_path: Union[Path, str] = None,
) -> AbsESPnetModel:
    """build espnet model"""
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args = argparse.Namespace(**args)
    model = ASRTask.build_model(args)

    return model


def load_param(
        model_path: Union[Path, str] = None,
) -> Any:
    """load model stated dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_state_dict = torch.load(model_path, map_location=device)
    return model_state_dict

