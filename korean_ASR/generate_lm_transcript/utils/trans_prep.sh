#!/usr/bin/env bash

# Copyright 2020 Electronics and Telecommunications Research Institute (Jeong-Uk, Bang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

export LC_ALL=C

. utils/parse_options.sh || exit 1;

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <db-dir> <text-dir>"
    echo "$#"
    echo "$0 $1 "
    echo "e.g.: $0 /mls/jubang/databases/KsponSpeech data/local/KsponSpeech"
    exit 1
fi

db=$1
text=$2
dataset=$3

tgt_case="df"
# fl: fluent transcription
# df: disfluent transcription
# dt: disfluent transcription with tag symbols ('/' or '+')

[ ! -f ${db} ] && echo "$0: no such file ${db}" && exit 1;
[ -f ${text}/.done ] && echo "$0: the KsponSpeech transcription exists ==> Skip" && exit 0;

mkdir -p ${text} ${text}/logs || exit 1;

# 1) get original transcription
[ ! -f "${db}" ] && echo "$0: no such transcription scripts/${dataset}.trn" && exit 1;
mkdir -p ${text} && cp -a ${db} ${text}/text.raw

# 2) get transcription files
echo "$0: get transcription files for KsponSpeech"
python utils/get_transcriptions.py --verbose 1 --clear \
    --type $tgt_case --log-dir ${text}/logs/get_transcription --unk-sym '[unk]' \
    --raw-trans ${text}/text.raw --out-fn ${text}/text.trn

echo "$0: successfully prepared transcription files for ${dataset} dataset"
touch ${text}/.done && exit 0;

