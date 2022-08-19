import os
import json
import pathlib
from pathlib import Path

from typing import Union
from typeguard import check_argument_types

# 자유대화 음성(일반남녀)
# https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=109
# |_ 자유대화음성
#     |_ 자유대화 음성(일반남녀)
#         |_ Training
#           |_  일반남여_[장소#1]
#               |_ 일반남여_일반통합_xxx_[장소#1]_xxx_00001.json
#               |_ 일반남여_일반통합_xxx_[장소#1]_xxx_00002.json
#               |_ 일반남여_일반통합_xxx_[장소#1]_xxx_00003.json
#               |_ ...
#           |_ 일반남여_[장소#2]
#               |_ 일반남여_일반통합_xxx_[장소#1]_xxx_00001.json
#               |_ 일반남여_일반통합_xxx_[장소#2]_xxx_00002.json
#               |_ ...
#        |_ Validation
#           |_ ...
#           |_ ...
#
# cat 자유대화음성/자유대화 음성(일반남녀)/Training/일반남여_[장소#1]/일반남여_일반통합_xxx_[장소#1]_xxx_00001.json
# {
#   "발화정보" : {
#     "stt" : "안녕하세요. 연습입니다.",
#     "scriptId" : "xxxxx",
#     "fileNm" : "일반남여_wwww_[장소]_xxxxx.wavp",
#     "recrdTime" : "5.000",
#     "recrdQuality" : "16K",
#     "recrdDt" : "xxxx-xx-xx xx:xx:xx",
#     "scriptSetNo" : "wwwww01"
#   },
#   "대화정보" : {
#     "recrdEnvrn" : "[장소]",
#     "colctUnitCode" : "[세부장소]",
#     "cityCode" : "[도시정보]",
#     "recrdUnit" : "[기록장비]",
#     "convrsThema" : "[대화주제]"
#   },
#   "녹음자정보" : {
#     "gender" : "여",
#     "recorderId" : "[recoder ID]",
#     "age" : [age]
#   }
# }

def prepare_text(
        base_dir: Union[str, pathlib.PosixPath, pathlib.WindowsPath],
        save_path: Union[str, pathlib.PosixPath, pathlib.WindowsPath]
):
    assert check_argument_types()

    json_category = '발화정보'
    json_sub_category = 'stt'

    flist = os.listdir(base_dir)
    with open(save_path, "w") as f:
        for file in flist:
            subdir = Path(base_dir, file)
            flist_2 = os.listdir(subdir)
            for file_2 in flist_2:
                subdir_2 = Path(subdir, file_2)
                flist_3 = os.listdir(subdir_2)
                for file_3 in flist_3:
                    subdir_3 = Path(subdir_2, file_3)
                    flist_4 = os.listdir(subdir_3)
                    for file_4 in flist_4:
                        subdir_4 = Path(subdir_3, file_4)
                        with subdir_4.open() as f_json:
                            json_data = json.load(f_json)
                        f.write(str(json_data[json_category][json_sub_category]) + '\n')



if __name__=="__main__":
    base_dir = "/media/user/NISP_HDD/NISP_HK/K_Lightweight_Conformer/AI_hub/KoreanFreeSpeech/"
    save_path = base_dir + "text"
    prepare_text(
        base_dir=base_dir,
        save_path=save_path
    )