import sys
import argparse

from korean_ASR.utils.set_logger import set_logger
from korean_ASR.interleaved_module.utils import model_synthesize


def get_parser():
    """Get default Arguments."""
    parser = argparse.ArgumentParser(description="""load model parameter for finetuning interleaved decoder.
                                                    The interleaved model is compound of conformer encoder and interleaved decoder.
                                                    The encoder is initialized with a pretrained conformer encoder.
                                                    The decoder is initialized with a pretrained LM""")
    # general configuration
    parser.add_argument(
        "--language-model-path",
        type=str,
        default="",
        help="""path to transformer lm model (.pth file path)""",
    )
    parser.add_argument(
        "--conformer-path",
        type=str,
        default="",
        help="""path to transformer conformer model (.pth file path)""",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="""path to save interleaved conformer model (.pth file path)""",
    )
    parser.add_argument(
        '--feats-stats',
        type=str,
        default='/home/ubuntu-1/Capstone/data/asr_config/asr_stats_raw_kr_bpe2309/train/feats_stats.npz',
        help=""""""
    )
    parser.add_argument(
        '--interleaved-config',
        type=str,
        default="",
        help="""config file path for constructing interleaved conformer model"""
    )
    parser.add_argument(
        '--bpe-model',
        type=str,
        default='/home/ubuntu-1/Capstone/data/kr_token_list/bpe_unigram2309/bpe.model',
        help=""""""
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()

    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        argv = [
            '--language-model-path', '/home/ubuntu-1/ASR/egs2/ksponspeech/exp/model/lm_model/valid.loss.ave_5best.pth',
            '--conformer-path', '/home/ubuntu-1/Capstone/data/asr_config/asr_train_asr_conformer_raw_kr_bpe2309/31epoch.pth',
            '--interleaved-config', '/home/ubuntu-1/ASR/custom_ASR/korean_ASR/interleaved/interleaved_conformer_config.yaml',
            '--save-path', './result'
        ]
        args = parser.parse_args(argv)

    logger = set_logger("{}/{}.log".format(args.save_path, __name__))
    model_synthesize(
        logger=logger,
        conformer_model_path=args.conformer_path,
        language_model_path=args.language_model_path,
        interleaved_config_path=args.interleaved_config,
        save_path=args.save_path,
    )


