import logging
import argparse

import torch
import numpy as np

from typing import Any
from typing import List

from pathlib import Path

from korean_ASR.label_refinary.module import FilteringLinearRegression
from korean_ASR.label_refinary.module import train, validate

from korean_ASR.label_refinary.utils.fileio import overwrite_orignal, save_label_filtered, read_dataset_details


def sortby_filtering_score(
        log_score: List[float] = None,
        normalized_filtering_score: List[float] = None,
        ground_truth: List[str] = None,
        prediction: List[str] = None,
        audio_file_name: List[str] = None,
        audio_path: List[str] = None,
) -> Any:
    """Returns transcript details sorted by normalized filtering score in ascending order.
       return data_sortedList[(
                    audio_file_name,
                    audio_file_path,
                    log_score,
                    normalized_filtering_score,
                    ground_truth,
                    predicted,
                    original_index)]"""

    data_dict = [(audio_file_name[idx],audio_path[idx], log_score[idx], normalized_filtering_score[idx], ground_truth[idx], prediction[idx], idx)
                                                                                                                        for idx in range(len(log_score))]
    data_sorted = sorted(data_dict, key=lambda item: item[3])

    return data_sorted


def calc_filtering_score(
        logger: logging.RootLogger,
        train_log_score: List[float],
        train_transcript: List[str],
        log_score: List[float],
        transcript: List[str],
        train_type: str,
        if_train: bool,
        epoch: int,
        learning_rate: float,
        save_path: str,
):
    """calculate filtering score and normalize with standard deviation
    1. filtering_score = (log_score - alpha x length - beta) / length
    2. normalized_filtering_score = filtering_score / np.std(filtering_score)
    Returns:
        normalized_filtering_score (List[float]) : filtering score normalized with standard deviation and length."""

    model_path = Path(f"{save_path}/model/{train_type}_regression.pth")
    model = FilteringLinearRegression()
    # If train option is set, train regression model. else load model from model_path.
    if if_train:
        logger.info(f"Traing regression moel with {len(train_transcript)} labels for {epoch} epochs...")
        model = train(
            model,
            log_score=train_log_score,
            script=train_transcript,
            epoch=epoch,
            learning_rate=learning_rate,
        )
		model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), model_path)
    else:
        logger.info(f"Regression loaded from: {model_path}")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    # Calculate filtering score for every transcript
    logger.info("Validating Filtering Score...")
    filtering_score = validate(
        model,
        log_score=log_score,
        script=transcript,
    )
    # normalize score with standard deviation
    normalized_filtering_score = filtering_score / np.std(filtering_score)

    return normalized_filtering_score


def label_filtering(args: argparse.Namespace, logger: logging.RootLogger) -> None:
    """Retrieve filtering score for every predicted label.
       The higher the score, the higher probabilty of the prediction being true.
       Filtering score below the filtering weight is removed from the transcript."""
    base_dir = Path(args.base_dir)
    decode_dir = args.decode_dir
    train_type = args.train_type
    if_train = args.if_train
    epoch = args.epoch
    learning_rate = args.learning_rate
    valid_type = args.valid_type
    filtering_weight = args.filtering_weight
    save_path = args.save_path
	learning_rate = args.learning_rate

    # For 'train' step, load LM fused log score & predicted text & ground truth text
    train_log_score, train_predicted = [], []
    if if_train:
        train_log_score, train_predicted, _, _, _ = read_dataset_details(base_dir, decode_dir, train_type)
    # For 'valid' step, load LM fused log score & predicted text & ground truth text
    log_score, predicted, ground_truth, audio_file_name, audio_file_path = read_dataset_details(base_dir, decode_dir, valid_type)

    # calculate normalized filtering score
    normalized_filtering_score = calc_filtering_score(
        logger,
        train_log_score,
        train_predicted,
        log_score,
        predicted,
        if_train=if_train,
        epoch=epoch,
        learning_rate=learning_rate,
        train_type=train_type,
        save_path=save_path
    )

    # Get transcript details sorted by normalized filtering score
    sorted_transcript_details = sortby_filtering_score(log_score, normalized_filtering_score, ground_truth, predicted, audio_file_name, audio_file_path)
    # length of transcripts bounded by filtering weight
    logger.info(f"Label Filtering...")
    bounded = len(normalized_filtering_score[normalized_filtering_score > filtering_weight])

    # filter out under bounded transcripts
    sorted_transcript_details = sorted_transcript_details[-1 * bounded:]
    # save filtered transcript
    save_label_filtered(
        logger=logger,
        save_path=save_path,
        data=sorted_transcript_details,
        base_dir=base_dir,
        decode_dir=decode_dir,
        dataset_type=valid_type,
        train_type=train_type,
        filtering_weight=filtering_weight,
    )
    # Transcripts with filtering score `higher` than filtering score is replaced by the model prediction
    # Transcripts with filtering score `below` filtering score is maintained with original transcript
    overwrite_orignal(
        logger=logger,
        save_path=save_path,
        base_dir=base_dir,
        dataset_type=valid_type,
        train_type=train_type,
        filtering_weight=filtering_weight,
    )
