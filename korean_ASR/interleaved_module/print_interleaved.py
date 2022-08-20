from typing import Union
from typing import Any

from pathlib import Path

from korean_ASR.utils.set_logger import set_logger
from korean_ASR.interleaved_module.utils import build_asr_model


def print_interleaved(
	logger: logger.RootLogger,
	model_path: Union[str, Path],
	config_path: Union[str, Path],
) -> Any:
	interleaved_param = torch.load(model_path)
	interleaved_conformer = build_asr_model(config_path)

	logger.info("Matching parameters to module.")
	interleaved_param.load_state_dict(interleaved_param)
	logger.info("All parameter is matched.")

	logger.info("")
	logger.info("Interleaved Module:")
	logger.info(interleaved_conformer)
	
def get_parser():
	parser = argparse.ArgumentParser()

	# general configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="""path to save interleaved conformer model (.pth file path)""",
    )
    parser.add_argument(
        '--interleaved-config',
        type=str,
        default="",
	)
	
	return parser


if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()
		
	logger = set_logger("../result/print_interleaved.log")

	print_interleaved(
		logger=logger,
		model_path=args.model_path,
		config_path=args.config_path,
	)
