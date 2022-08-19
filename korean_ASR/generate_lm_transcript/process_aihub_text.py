import sys
import argparse

from pathlib import Path

from korean_ASR.generate_lm_transcript.prepare_text.korean_free_speech import prepare_text as prepare_koreanfreespeech
from korean_ASR.generate_lm_transcript.prepare_text.korean_speech import prepare_text as prepare_koreanspeech

from korean_ASR.utils.set_logger import set_logger

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--korean-speech", type=str, required=True, help="path for korean speech")
    parser.add_argument("--korean-free-speech", type=str, required=True, help="path for korean free speech")
    parser.add_argument("--save-path", type=str, help="prepared text save path")
    return parser


if __name__ == "__main__":
    parser = get_parser()

    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        argv = [
            '--korean-speech',
            '/home/ubuntu-1/ASR/generate_lm_transcript/koreanspeech/data/data/remote/PROJECT/AI학습데이터/KoreanSpeech/data/2.Validation/1.라벨링데이터/',
            '--korean-free-speech', '/home/ubuntu-1/ASR/generate_lm_transcript/koreanfreespeech/',
            '--save-path', './'
        ]
        args = parser.parse_args(argv)

    logger = set_logger("{}/{}.log".format(args.save_path, __name__))

    args.save_path = Path(args.save_path)
    logger.info("Preparing korean_free_speech...")
    logger.info("saving in {}".format(args.save_path / "korean_free_speech.txt"))
    prepare_koreanfreespeech(
        save_path=args.save_path / "korean_free_speech.txt",
        base_dir=args.korean_free_speech,
    )
    logger.info("Preparing korean_speech...")
    logger.info("saving in {}".format(args.save_path / "korean_speech.txt"))
    prepare_koreanspeech(
        save_path=args.save_path / "korean_speech.txt",
        base_dir=args.korean_speech,
    )