#!/usr/bin/env bash

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

espnet_path=$("$python" ../utils/python-which.py espnet2)
espnet_task="$espnet_path/tasks/"


if [ ! -z "$(grep 'korean_ASR' $espnet_task/asr.py)" ]; then
    log "Interleaved Transformer is already supported."
else
	log "Including interleaved transformer supports..."
	sed 's/transformer=TransformerDecoder/transformer=TransformerDecoder,\n        interleaved=InterleavedTransformerDecoder/' \
        "$espnet_task/asr.py" > "$espnet_task/asr_modified.py"

	sed 's/frontend_choices =/from korean_ASR.interleaved_module.decoder.interleaved_transformer_decoder import InterleavedTransformerDecoder\n\nfrontend_choices =/' \
	   "$espnet_task/asr_modified.py" > "$espnet_task/asr_modified_2.py"

	sed 's/specaug=SpecAug,/specaug=SpecAug,\n        adaptive_specAug=AdaptiveSpecAug,/'\
		"$espnet_task/asr_modified_2.py" > "$espnet_task/asr_modified.py"

	sed 's/frontend_choices =/from korean_ASR.adaptive_specaug.adaptive_specaug import AdaptiveSpecAug\n\nfrontend_choices =/' \
	   "$espnet_task/asr_modified.py" > "$espnet_task/asr.py"

	log "Interleaved Transformer is supported."
fi

true

