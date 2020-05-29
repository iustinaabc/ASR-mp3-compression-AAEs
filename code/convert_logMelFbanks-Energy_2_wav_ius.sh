#!/bin/bash
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. utils/parse_options.sh || exit 1;

# feature reconstruction related
fs=16000      # sampling frequency
fmax=7800     # maximum frequency
fmin=80     # minimum frequency
n_mels=80     # number of mel basis
n_fft=256    # number of fft points
n_shift=160  # number of shift points
win_length=256 # window length
griffin_lim_iters=64  # number of Griffin-lim iterations (default=64)
nj=1
base_folder="et_en_raw_ONLY_fbanks_NO-cmvn"
#### IA:
# convert_fbank.sh usage: 
# $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
# e.g.: $0 data/train exp/griffin_lim/train wav
# Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data
# Options:
#   --nj <nj>                  # number of parallel jobs
#   --fs <fs>                  # sampling rate
#   --fmax <fmax>              # maximum frequency
#   --fmin <fmin>              # minimum frequency
#   --n_fft <n_fft>            # number of FFT points (default=1024)
#   --n_shift <n_shift>        # shift size in point (default=256)
#   --win_length <win_length>  # window length in point (default=)
#   --n_mels <n_mels>          # number of mel basis (default=80)
#   --iters <iters>            # number of Griffin-lim iterations (default=64)
#   --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.

# IA: <data-dir> is the folder containing the feats.scp file

# convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
#     ./data/et_en_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 \
#     ./data/et_en_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/speech_resynthesis_log \
#     ./data/et_en_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/resynthesized-speech_wavs
    # --fs ${fs} \
    # --fmax "${fmax}" \
    # --fmin "${fmin}" \
    # --n_fft ${n_fft} \
    # --n_shift ${n_shift} \
    # --win_length "${win_length}" \
    # --n_mels ${n_mels} \
    # --iters ${griffin_lim_iters} \
    #./data/et_en_raw_ONLY_fbanks-Librosa_NO-cmvn
    ### IA: --iters  (default=64)

        convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            --iters ${griffin_lim_iters} \
            ./data/${base_folder} \
            ./data/${base_folder}/speech_resynthesis_log3 \
            ./data/${base_folder}/resynthesized-speech_wavs3