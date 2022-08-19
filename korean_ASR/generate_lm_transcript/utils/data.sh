#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0
stage=1
stop_stage=100
#dataset
KoreanSpeech=
KoreanFreeSpeech=
save_path=

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

#. ./path.sh
#. ./cmd.sh


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data Preparation"
    for data in KoreanFreeSpeech KoreanSpeech; do
      log "Preparing ${data} text dataset"
      data_path=$(eval echo '${'$data'}')
      log "data_path: ${data_path}"
      ./utils/trans_prep.sh \
        ${data_path} \
        ${save_path}/local/${data} \
        ${data}
      ./utils/data_prep.sh \
        "${save_path}/local/${data}" \
        ${save_path}/${data}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"