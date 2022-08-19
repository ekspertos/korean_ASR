import sys
import argparse

from pathlib import Path

from korean_ASR.label_refinary.utils.types import str2bool
from korean_ASR.label_refinary.utils.label_scoring import label_filtering

from korean_ASR.utils.set_logger import set_logger


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
        "--filtering-weight",
        type=float,
        default=-0.5,
        help="""Filter transcripts scored below filtering-weight
                The higher the filtering score, the higher the accuracy of the prediction."""
    )
    parser.add_argument(
        "--if-train",
        type=str2bool,
        default=False,
        help="""If the linear regression model needs to be trained or not.
                Model is trained by train-type datset.""",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=70000,
        help="""Over-fitting the model does not degrade model performance."""
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="""Stochastic Gradient Descent optimizer learning rate."""
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./result/",
        help="""Stochastic Gradient Descent optimizer learning rate."""
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    argv = [
        '--train-type', 'dev',
        '--valid-type', 'eval_other',
        '--base-dir', 'G:\\내 드라이브\\espnet_2\\',
        '--decode-dir', 'backup\\asr_finetune_asr_conformer_without_encoder_update_seven_raw_kr_bpe2309\\decode_asr_beam_60_asr_model_valid.acc.ave',
        '--save-path', './result/',
        '--if-train', 'false',
        '--epoch', "0",
        '--filtering-weight', '0.5',
        '--learning-rate', '1e-3',
    ]
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    save_path = Path(f"{args.save_path}/{__name__}.log")
    logger = set_logger(save_path)
    label_filtering(args, logger)