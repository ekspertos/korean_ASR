import os
import sys
import logging
import argparse
import typeguard

from typing import Mapping
from typing import Union
from typing import List

import re
import pathlib
from pathlib import Path
from typeguard import check_argument_types

from util import NestedDictAction

from .set_logger import set_logger

"""

[Usage] comparing model outputs

>>> compare_result.py --log_filepath model_comparing.log
            --dataset "eval_other",
            --metric "score_cer",
            --base_dir", "xxx/espnet2/egs2/ksonspeech/",  # espnet egs2 base directory
            --evaluated_path", "first  = 'exp/xxx_xxx_bpe2309/www_www_valid.acc.ave'",    # first.valid.acc.ave file path from base_dir
            --evaluated_path", "second = 'exp/xxx_xxx_bpe2309/www_www_valid.acc.ave'",    # second.valid.acc.ave file path from base_dir
            --evaluated_path", "third  = 'exp/xxx_xxx_bpe2309/www_www_valid.acc.ave'",    # third.valid.acc.ave file path from base_dir
            --min_score 90
            
2022-08-13 17:29:44,979 [INFO] -------------------  comaparing model files   ------------------   
2022-08-13 17:29:44,980 [INFO] name: first
2022-08-13 17:29:44,980 [INFO] path: exp/xxx_xxx_bpe2309/www_www_valid.acc.ave
2022-08-13 17:29:45,234 [INFO] name: second
2022-08-13 17:29:45,235 [INFO] path: exp/xxx_xxx_bpe2309/www_www_valid.acc.ave
2022-08-13 17:29:45,491 [INFO] name: third
2022-08-13 17:29:45,492 [INFO] path: exp/xxx_xxx_bpe2309/www_www_valid.acc.ave
2022-08-13 17:29:45,763 [INFO] --------------------   ksponspeech_e0xxxx   --------------------   
2022-08-13 17:29:45,763 [INFO]      first : 90.25
2022-08-13 17:29:45,763 [INFO]     second : 91.33
2022-08-13 17:29:45,763 [INFO]      third : 89.90
2022-08-13 17:29:45,763 [INFO] 
2022-08-13 17:29:45,764 [INFO]  base text : KsponSpeech_E0xxxx 안녕하세요 예시입니다
2022-08-13 17:29:45,764 [INFO]      first : KsponSpeech_E0xxxx 안녕하세여 예시임니다
2022-08-13 17:29:45,764 [INFO]     second : KsponSpeech_E0xxxx 안녕하세유 예시입니다
2022-08-13 17:29:45,764 [INFO]      third : KsponSpeech_E0xxxx 안녕하세야 예시임다
2022-08-13 17:29:45,765 [INFO] --------------------   ksponspeech_e0xxxx   --------------------   
...
"""

def compare_model_output(
        logger: logging.RootLogger,
        base_dir: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
        evaluated_path: Mapping[str, str],
        dataset: str = "eval_other",
        metric: str = "score_cer",
        min_score: int = 90,
    ):
    assert check_argument_types()
    if type(base_dir) == str:
        filepath = Path(base_dir)

    name_list = []
    score_list = []
    text_list = []

    logger.info("-------------------  comaparing model files   ------------------   ")
    for idx, (name, subpath) in enumerate(evaluated_path.items()):
        logger.info("name: {}".format(name))
        logger.info("path: {}".format(subpath))

        with open(os.path.join(base_dir, subpath, dataset, "text"), encoding="utf-8") as f:
            text = f.readlines()
        with open(os.path.join(base_dir, subpath, dataset, metric, "result.txt"), encoding="utf-8") as f:
            score = f.readlines()

        name_list.append(name)
        score_list.append(score)
        text_list.append(text)

    with open(os.path.join(base_dir, "data", dataset, "text"),encoding="utf-8") as f:
        true_text = f.readlines()

    compare_score(name_list, score_list, text_list, true_text, min_score)


def compare_score(name_list: List[str], score_list: List[str], text_list: List[List[str]], true_text : List[str], min_score: int = 90):
    """score[0]
                     SYSTEM SUMMARY PERCENTAGES by SPEAKER
    ,--------------------------------------------------------------------------------------------------------------------------------------------------------------.
    |accmu_grad1_exp/asr_finetune_asr_conformer_without_encoder_update_seven_raw_kr_bpe2309/decode_asr_beam_60_asr_model_valid.acc.ave/eval_other/score_cer/hyp.trn|
    |--------------------------------------------------------------------------------------------------------------------------------------------------------------|
    |     SPKR                    |     # Snt          # Wrd      |     Corr              Sub             Del             Ins             Err           S.Err      |
    |-----------------------------+-------------------------------+------------------------------------------------------------------------------------------------|
    |     ksponspeech_e03001      |        1              30      |     83.3             16.7             0.0             3.3            20.0           100.0      |
    |-----------------------------+-------------------------------+------------------------------------------------------------------------------------------------|
    |     ksponspeech_e03002      |        1              35      |    100.0              0.0             0.0             0.0             0.0             0.0      |
    |-----------------------------+-------------------------------+------------------------------------------------------------------------------------------------|
    |     ksponspeech_e03003      |        1               8      |    100.0              0.0             0.0             0.0             0.0             0.0      |
    |-----------------------------+-------------------------------+------------------------------------------------------------------------------------------------|
    |     ksponspeech_e03004      |        1              36      |     94.4              2.8             2.8             2.8             8.3           100.0      |
    |-----------------------------+-------------------------------+------------------------------------------------------------------------------------------------|
    |     ksponspeech_e03005      |        1              19      |    100.0              0.0             0.0             0.0             0.0             0.0      |
    |-----------------------------+-------------------------------+------------------------------------------------------------------------------------------------|
    |     ksponspeech_e03006      |        1              43      |     97.7              0.0             2.3             0.0             2.3           100.0      |
    |-----------------------------+-------------------------------+------------------------------------------------------------------------------------------------|
    """

    category = {"SPKR":1, "Snt":2, "Wrd":3, "Corr":4, "Sub":5, "Del":6, "Ins":7, "Err":8, "S.Err":9}
    SPKR = category["SPKR"]
    Corr = category["Corr"]

    # stop after delimiter
    delimeter = "Sum/Avg"
    score_index = 0
    text_index = 0

    while True:
        model_score = re.split(r"\|| ", re.sub(r"[^0-9\w\.\/]+", " ", score_list[0][score_index], flags=re.I))
        # >>> model_score
        # ['', 'ksponspeech_e03001', '1', '30', '83.3', '16.7', '0.0', '3.3', '20.0', '100.0', '']
        score_index += 1

        # if blank line continue
        if len(model_score) <= 2:
            continue
        # if delimiter do break
        if model_score[1] == delimeter:
            break
        # if not contains score
        if len(model_score) < len(category):
            continue
        # if Corr score is not number
        try:
            _ = float(model_score[Corr])
        except:
            continue

        # prediction score & predicted text for all models
        correction_score_list = []
        pred_text_list = []
        for score, text in zip(score_list, text_list):
            correction_score = re.split(r"\|| ", re.sub(r"[^0-9\w\.\/]+", " ", score[score_index-1], flags=re.I))
            pred_text = text[text_index][:-1]
            correction_score_list.append(correction_score[Corr])
            pred_text_list.append(pred_text)

        text_index += 1
        # if first model and second model predicts the same result -> meaningless information
        if len(correction_score_list) > 1:
            if correction_score_list[0] == correction_score_list[1]:
                if float(correction_score_list[0]) > min_score:
                    continue

        speaker = model_score[SPKR]

        logger.info("--------------------   {}   --------------------   ".format(speaker))
        for name, pred_text, score in zip(name_list, pred_text_list, correction_score_list):
            logger.info("{0: >10} : {1:.2f}".format(name, float(score)))


        logger.info("")
        logger.info("{0: >10} : {1}".format("base text", true_text[text_index-1][:-1]))
        for name, pred_text, score in zip(name_list, pred_text_list, correction_score_list):
            logger.info("{0: >10} : {1}".format(name, pred_text))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_filepath', default='model_compared.log', help="output log to file")
    parser.add_argument('--evaluated_path', action=NestedDictAction, help="path from base_dir", default={})
    parser.add_argument('--base_dir', help="espnet egs2 path")
    parser.add_argument('--true_text_path')
    parser.add_argument('--min_score', default=90)
    parser.add_argument('--dataset', choices=['eval_other', 'eval_clean'], default='eval_other')
    parser.add_argument('--metric', choices=['score_cer', 'score_wer', 'score_swer'], default='score_cer')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    if len(sys.argv) != 1:
        args = parser.parse_args()
    else:
        argv = [
            "--log_filepath", "model_comparing.log",
            "--base_dir", "G:\\내 드라이브\\espnet_2\\",
            "--evaluated_path", "first='backup\\asr_finetune_asr_conformer_without_encoder_update_seven_raw_kr_bpe2309\\decode_asr_beam_60_asr_model_valid.acc.ave'",
            "--evaluated_path", "second='backup\\asr_finetune_asr_conformer_without_encoder_update_seven_raw_kr_bpe2309\\decode_asr_ctc_0.2_asr_model_valid.acc.ave'",
            "--evaluated_path", "third='backup\\asr_finetune_asr_conformer_without_encoder_update_seven_raw_kr_bpe2309\\decode_asr_ctc_0.2_asr_model_valid.acc.ave_5best'",
        ]
        args = parser.parse_args(argv)

    logger = set_logger(level=logging.INFO, filepath=args.log_filepath)

    compare_model_output(
        logger=logger,
        base_dir=args.base_dir,
        evaluated_path=args.evaluated_path,
        dataset=args.dataset,
        metric=args.metric,
    )