#! /usr/bin/env bash

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
	local fname=${BASH_SOURCE[1]##*/}
	echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

python=python3
script_abs_path=../utils/script/

korean_speech=/home/ubuntu-1/ASR/custom_ASR/korean_ASR/generate_lm_transcript/data/koreanspeech/data/data/remote/PROJECT/AI학습데이터/KoreanSpeech/data/2.Validation/1.라벨링데이터/
korean_free_speech=/home/ubuntu-1/ASR/custom_ASR/korean_ASR/generate_lm_transcript/data/koreanfreespeech/
save_path=./result/

. ${script_abs_path}/parse_options.sh

log "Generating clean transcript for training language model."
log "Dataset from aihub."
log "한국어음성: $korean_speech"
log "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=109"
log "자유음성(일반남녀): $korean_free_speech"
log "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=130"
log ""

${python} -m process_aihub_text \
  --korean-speech "$korean_speech" \
  --korean-free-speech "$korean_free_speech" \
  --save-path "$save_path"

log "Refining dataset.."
log ""

./utils/data.sh \
  --KoreanSpeech "$save_path/korean_free_speech.txt" \
  --KoreanFreeSpeech "$save_path/korean_speech.txt" \
  --save-path "$save_path"


log "Successfully finished. [elapsed=${SECONDS}s]"
