import logging
import argparse
import numpy as np
import torch

from shutil import rmtree
from pathlib import Path

from .filtering_module import FilteringLinearRegression, validate

from korean_ASR.label_refinary.utils.fileio import read_dataset_details
from korean_ASR.label_refinary.utils.fileio import save_trn
from korean_ASR.label_refinary.utils.matplotlib_plot import plot_cdf


def remove_path(path:str):
    path = Path(path)
    if path.exists():
        rmtree(path)


def filtering_module_details(args: argparse.Namespace, logger: logging.RootLogger) -> None:
    base_dir = Path(args.base_dir)
    decode_dir = args.decode_dir
    save_path = args.save_path
    valid_type = args.valid_type
    train_type = args.train_type
    filtering_weight_interval = args.filtering_weight_interval
    max_filtering_weight = args.max_filtering_weight
    min_filtering_weight = args.min_filtering_weight

    model_path = f"{save_path}/model/{train_type}_regression.pth"
    logger.info(f"loading model param from {model_path}")
    model = FilteringLinearRegression()
    model.load_state_dict(torch.load(model_path))

    # For 'valid' step, load LM fused log score & predicted text & ground truth text
    log_score, predicted, ground_truth, audio_file_name, audio_file_path = read_dataset_details(
                                                                                base_dir,
                                                                                decode_dir,
                                                                                valid_type
                                                                            )

    # Calculate filtering score for every transcript
    normalized_filtering_score = validate(
        model,
        log_score=log_score,
        script=predicted,
    )
    logger.info("Maximum Filtering Score: {}".format(max(normalized_filtering_score)))
    logger.info("Minimum Filtering Score: {}".format(min(normalized_filtering_score)))
    # plot cumulative distribution function
    plot_cdf(
        normalized_filtering_score,
        bins=300,
        save_path=save_path,
        dataset_type=valid_type,
        train_type=train_type,
    )

    # refresh directory
    remove_path(f"{save_path}/filtered_score/{train_type}/{valid_type}")
    # score cer, wer, swer by filtering weight
    for filtering_weight in np.arange(min_filtering_weight,
                                      max_filtering_weight+filtering_weight_interval,
                                      filtering_weight_interval):
        logger.info("Generating .trn file for filtering bound: {}".format(filtering_weight))
        save_trn(
            normalized_score=normalized_filtering_score,
            save_path=save_path,
            base_dir=base_dir,
            decode_dir=decode_dir,
            dataset_type=valid_type,
            train_type=train_type,
            filtering_weight=filtering_weight,
        )
