#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

fbankdir='./data/et_en_raw_ONLY_fbanks-Kaldi_NO-cmvn_updated_23.02.2020/features_NEW'
logdir='./data/et_en_raw_ONLY_fbanks-Kaldi_NO-cmvn_updated_23.02.2020/log_NEW'
#name='only_fbanks_espnet_librosa_noCMVN'
fs=16000
fmax=7800
fmin=80
n_mels=80
n_fft=512
n_shift=160
win_length=512
window=hann
write_utt2num_frames=true
cmd=run.pl
compress=true
normalize=16  # The bit-depth of the input wav files
filetype=mat

    make_fbank_librosa.sh --cmd "${train_cmd}" --nj 1 \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        ${write_num_frames_opt} \
        --compress ${compress} \
        --filetype ${filetype} \
        --normalize ${normalize} \
        ./data/et_en_raw_ONLY_fbanks-Kaldi_NO-cmvn_updated_23.02.2020 \
        ${logdir} \
        ${fbankdir}

# mkdir -p ${fbankdir} || exit 1;
# mkdir -p ${logdir} || exit 1;

# ${cmd} JOB=1:1 ${logdir}/${name}.JOB.log \
# compute-fbank-feats.py \
#           --fs ${fs} \
#           --fmax ${fmax} \
#           --fmin ${fmin} \
#           --n_fft ${n_fft} \
#           --n_shift ${n_shift} \
#           --win_length ${win_length} \
#           --window ${window} \
#           --n_mels ${n_mels} \
#           ${write_num_frames_opt} \
#           --compress ${compress} \
#           --filetype ${filetype} \
#           --normalize ${normalize} \
#           scp:./debugging_feat_extr_inversion/selected_wavs.scp \
#           ark,scp:${fbankdir}/${name}.ark,${fbankdir}/${name}.scp

###########################################################################

##### IA: attempt to redo the feature extraction with same params only for the data 
# . ./path.sh || exit 1;
# . ./cmd.sh || exit 1;

# # general configuration
# backend=pytorch
# stage=-1       # start from -1 if you need to start from data download
# stop_stage=100
# ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
# debugmode=1
# #dumpdir=dump   # directory to dump full features
# N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
# verbose=1      # verbose option
# resume=        # Resume the training from snapshot

# # feature extraction related 
# ###### IA: keep in mind that Librosa does not use the power of the spectrum when applying the Mel Fbanks - it just applies them on the amplitude!!
# fs=16000      # sampling frequency
# fmax=7800     # maximum frequency
# fmin=80       # minimum frequency
# n_mels=80     # number of mel basis
# n_fft=512    # number of fft points
# ### IA:  However, in speech processing, the recommended nfft value is 512, corresponding to 23 milliseconds at a sample rate of 22050 Hz. In any case, we recommend setting n_fft to a power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm.
# n_shift=160   # number of shift points ### IA: was originally 256
# #### IA: overlap of 10 ms is the default in Kaldi, which for us means 10*1e-3*16000 = 160 (points)
# win_length=512 # window length
# ### IA: in librosa.core.stft, If unspecified, defaults to win_length = n_fft
# ## https://librosa.github.io/librosa/generated/librosa.core.stft.html?highlight=stft#librosa.core.stft
# window=hann

# compress=true
# normalize=16  # The bit-depth of the input wav files
# filetype=mat

# # feature configuration
# do_delta=false
# cmvn=

# # train_config=conf/train_e4_subsamp12211_unit320_proj320_d1_300_mtlalpha0.3_epo20_ss-0.5_25.01.2020.yaml
# # decode_config=conf/decode_beam-sz10_ctcw-0.3_19.01.2020.yaml
# # advex_config=conf/advex_conf_bsz10_MovWin_eps0.3_winsz4_str2.yaml 

# # decoding parameter
# # recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
# # n_average=10 # ?????????
# #decode_advex=adv_ex
# #decode_model=decode
# #api=v1


# # data
# voxforge=downloads # original data directory to be stored
# lang=en # de, en, es, fr, it, nl, pt, ru

# # exp tag
# tag="" # tag for managing experiments.

# . utils/parse_options.sh || exit 1;

# # Set bash to 'debug' mode, it will exit on :
# # -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail

# # train_set=tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn
# # train_dev=dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn
# # recog_set="dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn et_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn"

# # if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
# #     echo "stage -1: Data Download"
# #     local/getdata.sh ${lang} ${voxforge}
# # fi

# # if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
# #     ### Task dependent. You have to make data the following preparation part by yourself.
# #     ### But you can utilize Kaldi recipes in most cases
# #     echo "stage 0: Data Preparation"
# #     selected=${voxforge}/${lang}/extracted
# #     # Initial normalization of the data
# #     local/voxforge_data_prep.sh ${selected} ${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn
# #     local/voxforge_format_data.sh ${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn
# # fi


# ############		IA: Wav -> mp3 -> wav conversion was done before running this script	########


# #feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
# #feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

# #if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#     ### Task dependent. You have to design training and dev sets by yourself.
#     ### But you can utilize Kaldi recipes in most cases
#     echo "stage 1: Feature Generation"
#     #bankdir=fbank
#     #fbankdir=fbank

#     #######     IA: usage of steps/make_fbank_pitch.sh :
#     # Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
#     # e.g.: $0 data/train
#     # Note: <log-dir> defaults to <data-dir>/log, and
#     # <fbank-dir> defaults to <data-dir>/data

#     # Generate the fbank features; by default 80-dimensional ONLY_fbanks-Librosa_ with pitch on each frame
#     # steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 15 --write_utt2num_frames true \
#         # data/all_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn exp/make_fbank/train_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn ${fbankdir}

#     # steps/make_fbank.sh --cmd "$train_cmd" --nj 15 --write_utt2num_frames true \
#     #     data/all_${lang}_raw_fbanks_NO-cmvn_06.02.2020 exp/make_fbank/train_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn ${fbankdir}

#      ##################      IA: Generate the fbank features with LIBROSA; by default 80-dimensional fbanks on each frame
# # Synthax of make_fbank_librosa.sh: 
# # Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
# # e.g.: $0 data/train exp/make_fbank/train mfcc
# # Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data
#     #fbankdir=fbank_librosa
#     make_fbank_librosa_ius.sh --cmd "${train_cmd}" --nj 1 \
#         --fs ${fs} \
#         --fmax "${fmax}" \
#         --fmin "${fmin}" \
#         --n_fft ${n_fft} \
#         --n_shift ${n_shift} \
#         --n_mels ${n_mels} \
#         --win_length "${win_length}" \
#         --window ${window} \
#         --compress ${compress} \
#         --filetype ${filetype} \
#         --normalize ${normalize} \
#         data/et_en_raw_ONLY_fbanks-Librosa_NO-cmvn_redo_21.02.2020 \
#         data/et_en_raw_ONLY_fbanks-Librosa_NO-cmvn_redo_21.02.2020/log_ius \
#         data/et_en_raw_ONLY_fbanks-Librosa_NO-cmvn_redo_21.02.2020/features_ius
#     # IA original line was: --win_length "${win_length}" \
#     # I selected win_length as 512 according to Kaldi
# # Syntax:
# #     #     ${datadir} \
# #     #     ${logdir} \
# #     #     ${fbankdir}

#     # utils/fix_data_dir.sh data/all_${lang}_raw_fbanks_NO-cmvn_06.02.2020
#     ## IA: --nj 10 was original
#     ### IA make_fbank_pitch.sh Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
#     #  e.g.: $0 data/train
#     # Note: <log-dir> defaults to <data-dir>/log, and
#     #       <fbank-dir> defaults to <data-dir>/data

#     # remove utt having more than 2000 frames or less than 10 frames or
#     # remove utt having more than 200 characters or 0 characters
#     # remove_longshortdata.sh data/all_${lang}_raw_fbanks_NO-cmvn_06.02.2020 data/all_trim_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn

#     # # following split consider prompt duplication (but does not consider speaker overlap instead)
#     # local/split_tr_dt_et.sh data/all_trim_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn \
#     #     data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn \
#     #     data/dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn \
#     #     data/et_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn
#     # rm -r data/all_trim_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn

#     # compute global CMVN
#     #compute-cmvn-stats scp:data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn/feats.scp data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn/cmvn.ark

#     # dump features for training
# #     if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
# #     utils/create_split_dir.pl \
# #         /export/b{14,15,16,17}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_set}/delta${do_delta}/storage \
# #         ${feat_tr_dir}/storage
# #     fi
# #     if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
# #     utils/create_split_dir.pl \
# #         /export/b{14,15,16,17}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_dev}/delta${do_delta}/storage \
# #         ${feat_dt_dir}/storage
# #     fi
# #     dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
# #         data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn/feats.scp data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
# #     dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
# #         data/dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn/feats.scp data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
# #     for rtask in ${recog_set}; do
# #         feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
# #         dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
# #             data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
# #             ${feat_recog_dir}
# #     done
# # fi