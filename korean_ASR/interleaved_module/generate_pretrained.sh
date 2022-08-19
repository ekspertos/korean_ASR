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

language_model_path=/home/ubuntu-1/ASR/egs2/ksponspeech/exp/model/lm_model/valid.loss.ave_5best.pth
conformer_path=/home/ubuntu-1/Capstone/data/asr_config/asr_train_asr_conformer_raw_kr_bpe2309/31epoch.pth
interleaved_config=./utils/interleaved_conformer_config.yaml
save_path=./result/

. ${script_abs_path}/parse_options.sh

log "Generating pretrained interleaved model."
log "Initialized with Conformer encoder and lm decoder."
log ""

${python} -m generate_module_pretrained \
  --language-model-path "$language_model_path" \
  --conformer-path "$conformer_path" \
  --interleaved-config "$interleaved_config" \
  --save-path "$save_path"

log "Successfully finished. [elapsed=${SECONDS}s]"
