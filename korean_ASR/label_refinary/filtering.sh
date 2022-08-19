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
score_opts=

# espnet egs2 ksponspeech base directory
base_dir=/home/ubuntu-1/ASR/egs2/ksponspeech
# relative path to model decoded directory
decode_dir=exp/model/decode_asr_beam_60_asr_model_valid.acc.ave/
save_dir=./result/

# dataset for training filtering model
train_type="dev"
# datatset to be filteree
valid_type="eval_other"

# linear regression model train config
if_train=false
epoch=1000
learning_rate=1e-3
filtering_weight=0.5

# parameter fro evaluting filtering model performance
min_weight=-4
max_weight=4
filtering_weight_interval=3

. "${script_abs_path}/parse_options.sh"

log "stage 1: Generate transcript with minimum filtering weight $filtering_weight"
${python} -m label_filtering \
	--base-dir "${base_dir}" --decode-dir "${decode_dir}" \
	--save-path "${save_dir}" --train-type "${train_type}" \
	--valid-type "${valid_type}" --filtering-weight "${filtering_weight}" \
	--if-train "${if_train}" --epoch "${epoch}" --learning-rate "${learning_rate}"


log "stage 2: Generate .trn files for each filtering bound."
log "Filtering bound is created with."
 log "minimum filtering weight: $min_weight."
 log "maximum filtering weight: $max_weight."
 log "weight interval: $filtering_weight_interval."
${python} -m prediction_rescored \
	--base-dir "${base_dir}" --decode-dir "${decode_dir}" \
	--save-path "${save_dir}" --train-type "${train_type}" \
	--valid-type "${valid_type}" \
	--filtering-weight-interval "${filtering_weight_interval}" \
	--max-filtering-weight "$max_weight" --min-filtering-weight "$min_weight"

log "stage 3: Score metric for each filtering bound."
for filter_weight in $(seq -f "%g" "$min_weight" "$filtering_weight_interval" "$max_weight"); do
	for _type in cer wer swer; do
		score_dir="${save_dir}/filtered_score/${train_type}/${valid_type}/filtered/${filter_weight}/score_${_type}/"
		sclite \
        	-r "${score_dir}/ref.trn" trn \
        	-h "${score_dir}/hyp.trn" trn \
        	-i rm -o all stdout > "${score_dir}/result.txt"

    	log "Write ${_type} result in ${score_dir}/result.txt"
    	grep -e Avg -e SPKR -m 2 "${score_dir}/result.txt"
	done
done

score_dir="${save_dir}/filtered_score/${train_type}/${valid_type}"
"${script_abs_path}./show_asr_result.sh" "${score_dir}"/filtered > "${score_dir}/RESULTS.md"


log "stage 4: Plot re-scored prediction for filtering weights."

${python} -m plot_rescored \
	--save-path "${save_dir}" \
	--train-type "${train_type}" \
	--valid-type "${valid_type}"

log "Successfully finished. [elapsed=${SECONDS}s]"
