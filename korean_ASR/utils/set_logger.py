
import sys
import logging

from typing import Union

from pathlib import Path
from pathlib import PosixPath, WindowsPath

from typeguard import check_argument_types




def set_logger(
        filepath: Union[str, PosixPath, WindowsPath]="logging.log",
        level: int=logging.INFO,
):
    assert check_argument_types()
    if type(filepath) == str:
        filepath = Path(filepath)

    filepath.parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(filepath, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    return logger