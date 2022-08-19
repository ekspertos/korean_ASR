import sys
import argparse

from pathlib import Path

from utils.types import str2bool
from utils.label_scoring import label_filtering

from korean_ASR.utils.set_logger import set_logger

from korean_ASR.label_refinary.utils.matplotlib_plot import plot_rescored_result


def get_parser():
    """Get default Arguments."""
    parser = argparse.ArgumentParser(description="Label Filtering Technique for ASR Enhancement")

    # general configuration
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

    return parser


if __name__ == '__main__':
    parser = get_parser()
    argv = [
        '--train-type', 'dev',
        '--valid-type', 'eval_other',
        '--save-path', './result/',
    ]
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    save_path = Path(f"{args.save_path}/log/{__name__}.log")
    logger = set_logger(save_path)
    plot_rescored_result(args, logger)
