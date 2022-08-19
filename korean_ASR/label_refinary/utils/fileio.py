import logging

import numpy as np

import re

from pathlib import PosixPath, WindowsPath
from pathlib import Path
from typing import List
from typing import Union

from typeguard import check_argument_types

from shutil import rmtree


# cer, wer swer
def save_trn(
        normalized_score: np.array,
        base_dir: Union[PosixPath, WindowsPath],
        decode_dir: str,
        filtering_weight: float,
        dataset_type: str,
        train_type: str,
        save_path: str,
    ):

    score_metrics = ['score_cer', 'score_wer', 'score_swer']

    decode_dir = base_dir / decode_dir / dataset_type
    not_filtered = normalized_score > filtering_weight

    filtering_weight = "%g" % filtering_weight
    save_path = Path(f"{save_path}/filtered_score/{train_type}/{dataset_type}/filtered/{filtering_weight}/")

    for score_metric in score_metrics:
        ref_path = decode_dir / score_metric / "ref.trn"
        hyp_path = decode_dir / score_metric / "hyp.trn"

        save_ref_path = save_path / score_metric / "ref.trn"
        save_hyp_path = save_path / score_metric / "hyp.trn"

        save_ref_path.parent.mkdir(exist_ok=True, parents=True)

        with ref_path.open(encoding='utf-8') as f:
            ref = f.readlines()
        with hyp_path.open(encoding='utf-8') as f:
            hyp = f.readlines()

        ref_filtered = [ref[idx] for idx, c in enumerate(not_filtered) if c == True]
        hyp_filtered = [hyp[idx] for idx, c in enumerate(not_filtered) if c == True]

        with save_ref_path.open("w", encoding='utf-8')  as f:
            f.writelines(ref_filtered)
        with save_hyp_path.open("w", encoding='utf-8') as f:
            f.writelines(hyp_filtered)



def save_label_filtered(
        logger: logging.RootLogger,
        save_path: str,
        data: List[int],
        base_dir: Union[PosixPath, WindowsPath],
        decode_dir: str,
        dataset_type: str,
        train_type: str,
        filtering_weight: int,
):
    # save renewed transcript path
    save_path = Path(f"{save_path}/filtered_transcript/{train_type}/{dataset_type}/{filtering_weight}/filtered")
    # model decoded directory path
    decode_dir = base_dir / decode_dir / dataset_type
    # transcript ground truth path
    predicted_text_path = decode_dir / "text"
    save_path = save_path / "text"

    data_idx = [a[-1] for a in data]

    with predicted_text_path.open(encoding='utf-8') as f:
        predicted = f.readlines()

    logger.info(f"Predicted Dataset: {predicted_text_path} ({len(predicted)})")
    _text = [predicted[idx] for idx in data_idx]

    logger.info(f"Filtered Prediction {save_path} ({len(_text)})")
    save_path.parent.mkdir(exist_ok=True, parents=True)
    with save_path.open("w",encoding="utf-8") as f:
        f.writelines(_text)



# lead_data
def read_dataset_details(
        base_dir: Union[PosixPath, WindowsPath],
        decode_dir: str,
        dataset_type: str,
    ):
    """
    Args:
        base_dir: espnet egs2 base directory
        decode_dir: model decode directory (relative directory path from base_dir)
        dataset_type: import dataset type (choices=['train', 'dev', 'eval_clean', 'eval_other'])

    Returns:
        log_score: transcript decoded to LM fused log score
        transcript: predicted transcript from speech input
        ground_truth: ground truth transcript for speech input
        audio_file_name: audio file name
                        ['KsponSpeech_E03001', 'KsponSpeech_E03002', 'KsponSpeech_E03003',..]
    """

    assert check_argument_types()

    # model decoded directory path
    decode_dir = base_dir / decode_dir / dataset_type
    # transcript ground truth path
    ground_truth_dir = base_dir / "data" / dataset_type

    # LM fused log score path
    score_path = decode_dir / "score"
    # predicted transcript path
    transcript_path = decode_dir / "text"
    # paired transcript with audio path
    wav_scp_path = ground_truth_dir / "tmp/pcm.scp"
    # ground truth text path
    ground_truth_path = ground_truth_dir / "text"

    with score_path.open(encoding='utf-8') as f:
        score = f.readlines()
    log_score = [float(re.split(" |\(|\)", s)[2]) for s in score]
    log_score = np.array(log_score)

    audio_file_name = [re.split(" |\(|\)", s)[0] for s in score]
    audio_file_name = np.array(audio_file_name)

    with transcript_path.open(encoding='utf-8') as f:
        text = f.readlines()
    transcript = [" ".join(re.split(" |\\n",s)[1:-1]) for s in text]
    transcript = np.array(transcript)

    with ground_truth_path.open(encoding='utf-8') as f:
        ground_truth = f.readlines()
    ground_truth = [" ".join(re.split(" |\\n",s)[1:-1]) for s in ground_truth]
    ground_truth = np.array(ground_truth)

    with wav_scp_path.open(encoding='utf-8') as f:
        wav_scp = f.readlines()
    audio_path = ([re.split(" |\\n", s)[1] for s in wav_scp])

    return log_score, transcript, ground_truth, audio_file_name, audio_path

def overwrite_orignal(
        logger: logging.RootLogger,
        save_path: str,
        base_dir: Union[PosixPath, WindowsPath],
        dataset_type: str,
        train_type: str,
        filtering_weight: int,
    ):
    """The filtered transcript is overwritten with the original labels"""
    # transcript ground truth path
    ground_truth_path = base_dir / "data" / dataset_type / "text"
    # filtered_text generated by save_data function
    filtered_text_path = Path(f"{save_path}/filtered_transcript/{train_type}/{dataset_type}/{filtering_weight}/filtered/")
    filtered_text_path = filtered_text_path / "text"
    save_path = Path(f"{save_path}/filtered_transcript/{train_type}/{dataset_type}/{filtering_weight}/full/")
    save_path = save_path / "text"

    with ground_truth_path.open(encoding='utf-8') as f:
        ground_truth = f.readlines()
    with filtered_text_path.open(encoding='utf-8') as f:
        filtered_text = f.readlines()

    filtered_text_dict = {re.split(" ", s)[0]:s for s in filtered_text}
    ground_truth_dict = {re.split(" ", s)[0]:s for s in ground_truth}
    # >>> len(ground_truth), len(overwrite_text)
    # Out[1]: (619504, 427000)

    # If transcript is removed in filtered text, use original transcript
    overwritten_text = [filtered_text_dict[k] if k in filtered_text_dict else ground_truth_dict[k] for k in ground_truth_dict]

    logger.info(f"Removed transcript is replaced by the orginal labels: {save_path} ({len(ground_truth) - len(filtered_text)}")
    # save full version of ksponspeech transcript
    save_path.parent.mkdir(exist_ok=True, parents=True)
    with save_path.open("w",encoding='utf-8') as f:
        f.writelines(overwritten_text)

    print()
