
import os
import json
from typing import Union
import pathlib
from pathlib import Path

from typeguard import check_argument_types

# 한국인 대화 음성
# https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=130
# |_ KoreanSpeech
#     |_ data
#        |_ 1.Training
#            |_ 1.라벨링데이터
#               |_ 1.방송
#                  |_ broadcast_01
#                     |_ hobby_01_scripts.txt
#                     |_ hobby_01_metadata.txt
#                  |_ broadcast_02
#                  |_ ...
#               |_ 2.취미
#                  |_ ...
#           |_ 1.원천데이터
#               |_ ...
#        |_ 2.Validation
#           |_ ...
#           |_ ...
#
# cat KoreanSpeech/data/1.Training/1.라벨링데이터/hobby_01_scripts.txt/hobby_01_scripts.txt
# /2.Validation/2.원천데이터/2.취미/hobby_01/001/hobby_xxxxxxxx.wav :: text_1
# /2.Validation/2.원천데이터/2.취미/hobby_01/001/hobby_xxxxxxxx.wav :: text_2
# ...

def prepare_text(
        base_dir: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
        save_path: Union[str, pathlib.PosixPath, pathlib.WindowsPath]
):
    assert check_argument_types()

    category = ["1.방송", "2.취미", "3.일상안부", "4.생활", "5.날씨", "6.경제", "7.놀이", "8.쇼핑"]
    #o_category = ["broadcast","hobby", "dialog","life", "weather", "economy", "play", "shopping"]

    with open(save_path, "w") as f_write:
        for idx, cat in enumerate(category):
            subdir = Path(base_dir, cat)
            flist = os.listdir(subdir)
            for file in flist:
                subdir_2 = Path(subdir, file)
                flist_2 = os.listdir(subdir_2)
                for file_2 in flist_2:
                    subdir_3 = Path(subdir_2, file_2)
                    if file_2.endswith("scripts.txt"):
                        with subdir_3.open() as f:
                            text = f.readlines()
                        s_pos = text[10].find("::") + 3
                        text = [t[s_pos:] for t in text]
                        f_write.writelines(text)


if __name__=="__main__":
    base_dir = "/media/user/NISP_HDD/NISP_HK/K_Lightweight_Conformer/AI_hub/KoreanSpeech/1.Training/1.라벨링데이터/"
    prepare_text(base_dir)