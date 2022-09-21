#!/usr/bin/env bash

# This script computes audio features using kaldi on input audio files. The
# generated features can be used as reference data or input to other test cases.

# options:
# - e: exit immediately when a pipeline fails
# - u: treat unset variables as an error
# - o pipefail: return value of rightmost command in a pipeline
set -euo pipefail

# Get the top level directory.
TOP=$(git rev-parse --show-toplevel)

# Kaldi repo directory with built binaries.
kaldi="/home/$USER/git/kaldi"

# ---------------------------------

function write_feat_conf() {
    out_dir=$1
    sample_freq=$2
    num_mels=$3
    snip_edges=$4
    frame_length=$5
    frame_shift=$6
    raw_energy=$7
    num_ceps=$8
    use_energy=$9

    cat <<EOF > ${out_dir}/mfcc.conf
        --sample-frequency=${sample_freq}
        --frame-length=${frame_length}
        --frame-shift=${frame_shift}
        --use-energy=${use_energy}
        --raw-energy=${raw_energy}
        --dither=0
        --low-freq=20
        --high-freq=-400
        --num-mel-bins=${num_mels}
        --num-ceps=${num_ceps}
        --snip-edges=${snip_edges}
EOF

    cat <<EOF > ${out_dir}/fbank.conf
        --sample-frequency=${sample_freq}
        --frame-length=${frame_length}
        --frame-shift=${frame_shift}
        --raw-energy=${raw_energy}
        --dither=0
        --low-freq=20
        --high-freq=-400
        --num-mel-bins=${num_mels}
        --use-log-fbank=true
        --use-power=true
        --snip-edges=${snip_edges}
EOF
}

function generate_feats() {
    out_dir=$1
    audio_path=$2
    input_scp=$3

    ln -sf "$(realpath ${audio_path} --relative-to=${out_dir})" "${out_dir}/audio.wav"

    # Computing Filter bank features.
    "${kaldi}/src/featbin/compute-fbank-feats" \
        --config="${out_dir}/fbank.conf" \
        scp,p:<(echo "${input_scp}") \
        ark,t:"${out_dir}/fbank.ark.txt"

    # Computing MFCCs.
    "${kaldi}/src/featbin/compute-mfcc-feats" \
        --config="${out_dir}/mfcc.conf" \
        scp,p:<(echo "${input_scp}") \
        ark,t:"${out_dir}/mfcc.ark.txt"
}

function write_cmvn_conf() {
    out_dir=$1
    window=$2
    norm_vars=$3
    center=$4
    min_window=$5

    cat <<EOF > ${out_dir}/cmvn.conf
        --cmn-window=${window}
        --norm-vars=${norm_vars}
        --center=${center}
        --min-cmn-window=${min_window}
EOF
}

function apply_cmvn() {
    out_dir=$1
    mfcc_dir=$2

    ln -sf "$(realpath ${mfcc_dir}/mfcc.ark.txt --relative-to=${out_dir})" "${out_dir}/mfcc.ark.txt"

    # Applying CMVN.
    "${kaldi}/src/featbin/apply-cmvn-sliding" \
        --config="${out_dir}/cmvn.conf" \
        ark,t:"${out_dir}/mfcc.ark.txt" \
        ark,t:"${out_dir}/cmvn.ark.txt"
}

function write_vad_conf() {
    out_dir=$1
    energy_threshold=$2
    mean_scale=$3
    frames_context=$4
    proportion_threshold=$5

    cat <<EOF > ${out_dir}/vad.conf
        --vad-energy-threshold=$2
        --vad-energy-mean-scale=$3
        --vad-frames-context=$4
        --vad-proportion-threshold=$5
EOF
}

function compute_vad() {
    out_dir=$1
    mfcc_dir=$2

    ln -sf "$(realpath ${mfcc_dir}/mfcc.ark.txt --relative-to=${out_dir})" "${out_dir}/mfcc.ark.txt"

    # Computing VAD.
    "${kaldi}/src/ivectorbin/compute-vad" \
        --config="${out_dir}/vad.conf" \
        ark,t:"${out_dir}/mfcc.ark.txt" \
        ark,t:"${out_dir}/vad.ark.txt"
}


function write_delta_conf() {
    out_dir=$1
    order=$2
    window=$3

    cat <<EOF > ${out_dir}/delta.conf
        --delta-order=${order}
        --delta-window=${window}
EOF
}

function compute_deltas() {
    out_dir=$1
    mfcc_dir=$2

    ln -sf "$(realpath ${mfcc_dir}/mfcc.ark.txt --relative-to=${out_dir})" "${out_dir}/mfcc.ark.txt"

    # Computing Deltas.
    "${kaldi}/src/featbin/add-deltas" \
        --config="${out_dir}/delta.conf" \
        ark,t:"${out_dir}/mfcc.ark.txt" \
        ark,t:"${out_dir}/deltas.ark.txt"
}

use_energy=true
n=0

# for sample_freq in 16000; do
#     for num_mels in 23 30; do
#         for snip_edges in true false; do
#             for frame_length in 25 32; do
#                 for frame_shift in 10 16; do
#                     for raw_energy in true false; do
#                         for num_ceps in 23 30; do
#                             if [[ ${num_ceps} -gt ${num_mels} ]]; then
#                                 continue
#                             fi

#                             n=$((n+1))
#                             audio_path="${TOP}/kaldi_tflite/lib/testdata/librispeech_2_trimmed.wav"
#                             input_scp="seg1 ${audio_path}"

#                             # Output directory where reference input and outputs will be written to.
#                             id=$(printf "%03d" ${n})
#                             out_dir="${TOP}/kaldi_tflite/lib/testdata/feats/src/fbank_mfcc/${sample_freq}_${id}"
#                             mkdir -p "${out_dir}"

#                             write_feat_conf "${out_dir}" ${sample_freq} ${num_mels} ${snip_edges} ${frame_length} ${frame_shift} ${raw_energy} ${num_ceps} ${use_energy}
#                             generate_feats "${out_dir}" "${audio_path}" "${input_scp}"
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# # A couple more unique combinations.
# use_energy=false
# sample_freq=16000
# snip_edges=true
# frame_length=25
# frame_shift=10

# for raw_energy in true false; do
#     for num_mels in 23 30; do
#         for num_ceps in 23 30; do
#             if [[ ${num_ceps} -gt ${num_mels} ]]; then
#                 continue
#             fi
#             n=$((n+1))
#             audio_path="${TOP}/kaldi_tflite/lib/testdata/librispeech_2_trimmed.wav"
#             input_scp="seg1 ${audio_path}"

#             # Output directory where reference input and outputs will be written to.
#             id=$(printf "%03d" ${n})
#             out_dir="${TOP}/kaldi_tflite/lib/testdata/feats/src/fbank_mfcc/${sample_freq}_${id}"
#             mkdir -p "${out_dir}"

#             write_feat_conf "${out_dir}" ${sample_freq} ${num_mels} ${snip_edges} ${frame_length} ${frame_shift} ${raw_energy} ${num_ceps} ${use_energy}
#             generate_feats "${out_dir}" "${audio_path}" "${input_scp}"
#         done
#     done
# done

# # Applying CMVN with various conifgs on a single set of MFCCs.
# mfcc_dir="${TOP}/kaldi_tflite/lib/testdata/feats/src/fbank_mfcc/16000_001"
# n=0
# for center in true; do
#     for min_window in 100; do
#         for window in 199 200 201 600; do
#             for norm_vars in false true; do
#                 n=$((n+1))
#                 id=$(printf "%03d" ${n})
#                 out_dir="${TOP}/kaldi_tflite/lib/testdata/feats/src/cmvn/16000_001_${id}"
#                 mkdir -p "${out_dir}"

#                 write_cmvn_conf "${out_dir}" ${window} ${norm_vars} ${center} ${min_window}
#                 apply_cmvn "${out_dir}" "${mfcc_dir}"
#             done
#         done
#     done
# done

# # Computing VAD with various conifgs on a single set of MFCCs.
# mfcc_dir="${TOP}/kaldi_tflite/lib/testdata/feats/src/fbank_mfcc/16000_001"
# n=0
# for energy_threshold in 5.0 10.0; do
#     for mean_scale in 0.0 0.5 1.0; do
#         for frames_context in 0 1 2; do
#             for proportion_threshold in 0.1 0.6 0.9; do
#                 n=$((n+1))
#                 id=$(printf "%03d" ${n})
#                 out_dir="${TOP}/kaldi_tflite/lib/testdata/feats/src/vad/16000_001_${id}"
#                 mkdir -p "${out_dir}"

#                 write_vad_conf "${out_dir}" ${energy_threshold} ${mean_scale} ${frames_context} ${proportion_threshold}
#                 compute_vad "${out_dir}" "${mfcc_dir}"
#             done
#         done
#     done
# done

# Computing Deltas with various conifgs on a single set of MFCCs.
mfcc_dir="${TOP}/kaldi_tflite/lib/testdata/feats/src/fbank_mfcc/16000_001"
n=0
for order in 1 2 3; do
    for window in 1 3 8; do
        n=$((n+1))
        id=$(printf "%03d" ${n})
        out_dir="${TOP}/kaldi_tflite/lib/testdata/feats/src/deltas/16000_001_${id}"
        mkdir -p "${out_dir}"

        write_delta_conf "${out_dir}" ${order} ${window}
        compute_deltas "${out_dir}" "${mfcc_dir}"
    done
done
