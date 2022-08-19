import sys
import argparse

from pathlib import Path

from utils.types import str2bool
from utils.label_scoring import label_filtering

from korean_ASR.utils.set_logger import set_logger

from korean_ASR.label_refinary.module.filtering_module_details import filtering_module_details


def get_parser():
    """Get default Arguments."""
    parser = argparse.ArgumentParser(description="Label Filtering Technique for ASR Enhancement")

    # general configuration
    parser.add_argument(
        "--base-dir",
        type=str,
        default="~/espnet/egs2/ksponspeech/asr1/",
        help="""Directory path to KsponSpeech ASR directory.
                The directory path looks like
                <home>/espnet/egs2/ksponspeech/asr1/""",
    )
    parser.add_argument(
        "--decode-dir",
        type=str,
        default="exp/asr_train_asr_conformer8_n_fft512_hop_length256_raw_kr_bpe2309/asr_train_asr_conformer8_n_fft512_hop_length256_raw_kr_bpe2309_valid.acc.best/",
        help="""Relative path from ESPNet ASR directory to model decode result path.
                "/<base-dir>/<exp-dir>/<train-config>/<decoded-result>"
                |_ eval_other
                    |_ score_cer
                        |_ result.txt
                |_ eval_clean""",
    )
    parser.add_argument(
        "--train-type",
        type=str,
        default="dev",
        help="""Dataset for training the linear regression model.
                The linear regression model is used for data filtering.
                prediction score is calculated using `log-probability` and `transcript length`.
                S(logProb,length) = (logProb - alpha * length) / (sigma * sqrt(length))""",
    )
    parser.add_argument(
        "--valid-type",
        type=str,
        default="train",
        help="""Dataset to be filtered.
                The transcripts is scored by the linear regression model.
                Out-of-bound transcripts is filtered using filtering-weight.""",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./result/",
        help="""Stochastic Gradient Descent optimizer learning rate."""
    )
    parser.add_argument(
        "--filtering-weight-interval",
        type=float,
        default=0.25,
        help="""The filtered transcript is bounded using filtering-score bound. 
                    The filtered transcript is generated for every filtering score interval
                    Starting with -4 and ending with 4,
                    If the interval is 0.25 the transcript is generated for [-0.4, -0.375, -0.35 ... 0.35, 0.375, 0.4]"""
    )
    parser.add_argument(
        "--max-filtering-weight",
        type=float,
        default=4,
        help="""Maximum filtering weight for filtering transcript.
                    The filtered transcript is re-scored."""
    )
    parser.add_argument(
        "--min-filtering-weight",
        type=float,
        default=-4,
        help="""Minimum filtering weight for filtering transcript.
                    The filtered transcript is re-scored."""
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    argv = [
        '--train-type', 'dev',
        '--valid-type', 'eval_other',
        '--base-dir', 'G:\\내 드라이브\\espnet_2\\',
        '--decode-dir', 'backup\\asr_finetune_asr_conformer_without_encoder_update_seven_raw_kr_bpe2309\\decode_asr_beam_60_asr_model_valid.acc.ave',
        '--save-path', './result/',
        '--filtering-weight-interval', '0.25',
    ]
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    save_path = Path(f"{args.save_path}/log/{__name__}.log")
    logger = set_logger(save_path)
    filtering_module_details(args, logger)
