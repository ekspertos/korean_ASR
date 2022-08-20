#!/usr/bin/env bash

# Copyright 2020 Electronics and Telecommunications Research Institute (Jeong-Uk, Bang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <src-dir> <dst-dir>"
    echo "e.g.: $0 data/local/KsponSpeech data/KoreanSpeech"
    exit 1
fi

src=$1
dst=$2

data=$(echo $dst | sed 's:\.:/:' | awk -v src=$src -F"/" '{print src"/text.trn"}')
temp=tmp

mkdir -p ${dst} ${dst}/$temp || exit 1;

[ ! -d ${db} ] && echo "$0: no such directory ${db}" && exit 1;
[ ! -f ${data} ] && echo "$0: no such file ${data}. please re-run the script of 'local/trans_prep.sh'." && exit 1;

text=${dst}/text; [[ -f "${text}" ]] && rm ${text}

echo "data: $data"

# 3) prepare text
cat $data | cut -d' ' -f3- > ${dst}/${temp}/text.org
cat ${dst}/${temp}/text.org | utils/lowercase.perl | utils/remove_punctuation.pl > ${dst}/text


ntext=$(wc -l <$text)

echo "$0:"
echo "$0: ====== successfully prepared dataset in ======"
echo "$0: ====== ${dst} ======"
echo "$0:"
exit 0;
