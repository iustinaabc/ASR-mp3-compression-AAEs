#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
### IA: We attempt to take the network trained on Librosa ONLY fbank feats from raw audio and make it recognize NEWLY reconstructed audio as was created by the script reconstruct_et_raw_ONLY_fbanks-Librosa_noCMVN_ius.sh 

# general configuration
# date_advex=25.02.2020
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
#dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature extraction related 
###### IA: keep in mind that Librosa does not use the power of the spectrum when applying the Mel Fbanks - it just applies them on the amplitude!!
fs=16000      # sampling frequency
fmax=7800     # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=512    # number of fft points
### IA:  However, in speech processing, the recommended nfft value is 512, corresponding to 23 milliseconds at a sample rate of 22050 Hz. In any case, we recommend setting n_fft to a power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm.
n_shift=160   # number of shift points ### IA: was originally 256
#### IA: overlap of 10 ms is the default in Kaldi, which for us means 10*1e-3*16000 = 160 (points)
win_length=512 # window length
### IA: in librosa.core.stft, If unspecified, defaults to win_length = n_fft
## https://librosa.github.io/librosa/generated/librosa.core.stft.html?highlight=stft#librosa.core.stft
window=hann

compress=true
normalize=16  # The bit-depth of the input wav files
filetype=mat

# feature configuration
do_delta=false
cmvn=

#train_config=conf/train_e4_subsamp12211_unit320_proj320_d1_300_mtlalpha0.3_epo20_ss-0.5_25.01.2020.yaml
decode_config=conf/decode_batch-sz100_beam-sz20_ctcw-0.3_12.03.2020.yaml
#conf/decode_beam-sz10_ctcw-0.3_19.01.2020.yaml
#advex_config=conf/advex_conf_bsz10_MovWin_eps0.3_winsz4_str2.yaml 

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10 # ?????????
#decode_advex=adv_ex
#decode_model=decode
#api=v1


# data
voxforge=downloads # original data directory to be stored
lang=en # de, en, es, fr, it, nl, pt, ru

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#train_set=tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020
#train_dev=dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020
#recog_set="dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 et_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020"

#advEx_set="et_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020"

### IA: ACHTUNG: the Kaldi-style files from  data/$recog_set have to be manually adjusted to contain the path to the new RECONSTRUCTED audio
## specifically, wav.scp has to be modified (possibly the text binary file with the transcription might need modification too!)
### Be careful about the flac files which were there originally 


# Synthax of make_fbank_librosa.sh: 
# Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
# e.g.: $0 data/train exp/make_fbank/train mfcc
# Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data

data_types="compr_128kbps"
# "compr_128kbps compr_64kbps compr_24kbps"
# "raw compr_128kbps compr_64kbps compr_24kbps"

### IA: Before running this script, the folders with the compressed and RECONSTRUCTED data (eg. data/en_compr_128kbps_reconstructed) have to contain the etc folders (they have to be manually copied from data/en folder )
for data_type in $data_types; do

    recog_set=et_en_${data_type}_ONLY_fbanks-Librosa_NO-cmvn_from_RECONSTR_audio_19.03.2020
    ### IA: I manually created the above folders for all the 3 compression rates...and I copied the wav.scp from the folder all_et_en_raw_ONLY_fbanks-Librosa_NO-cmvn_from_reconstr_audio_ius_28.02.2020_2 (which was also manually created), and then just modified the wav.scp to have the right paths of the compressed & RECONSTRUCTED files (with sed - substitute command)
    
    if [ $data_type == compr_128kbps ]; then
        expdir=exp/tr_en_compr_128kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo25_ss0.5_28.02.2020
    elif [ $data_type == raw ]; then
        expdir=exp/tr_en_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_23.02.2020
        
        recog_set=all_et_en_raw_ONLY_fbanks-Librosa_NO-cmvn_from_reconstr_audio_ius_28.02.2020_2
    elif [ $data_type == compr_64kbps ]; then
        expdir=exp/tr_en_compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_24.02.2020
    elif [ $data_type == compr_24kbps ]; then
        expdir=exp/tr_en_compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_26.02.2020
    fi

    echo -e "      \n Decoding reconstructed audio from en_${data_type}_reconstructed with the model from $expdir "
    echo -e "       recog_set folder is in data/$recog_set"

    # if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    #     ### Task dependent. You have to make data the following preparation part by yourself.
    #     ### But you can utilize Kaldi recipes in most cases
    #     echo "stage 0: Data Preparation" 
    #     selected=${voxforge}/${lang}_${data_type}_reconstructed/extracted
    #     # Initial normalization of the data
    #     local/voxforge_data_prep.sh ${selected} $recog_set
    #     local/voxforge_format_data.sh $recog_set
    # fi

    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        echo "stage 1: Extract new features from the RECONSTRUCTED ${data_type} audio TEST set"
        # fbankdir=fbank_librosa
        make_fbank_librosa.sh --cmd "${train_cmd}" --nj 15 \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --window ${window} \
            --n_mels ${n_mels} \
            data/$recog_set \
            data/$recog_set/log \
            data/$recog_set/features
    fi
        ####        IA:
        # the features from the reconstructed RAW audio were written in all_$recog_set folder 
        # for the compressed samples, I removed the "all_"

    dict=data/lang_1char_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/tr_${lang}_units.txt
    ## We take the dictionary computed originally 
    # Build the json file from the new features
    # Usage of data2json.sh: $0 <data-dir> <dict>
    # e.g. $0 data/train data/lang_1char/train_units.txt

    data2json.sh --lang ${lang} --feat data/$recog_set/feats.scp \
            data/$recog_set ${dict} > data/$recog_set/data.json
    # IA: original was all_$recog_set

    # NExt, we skip utils/fix_data_dir.sh, remove_longshort.sh and split_tr_dt_et.sh stages 
    # We also skip dictionary generation and network training

    # We go straight to the decoding of the new features
    #expdir=exp/decode_raw_et_RECONSTR_audio_from_Librosa_fbanks_28.02.2020
    #mkdir -p ${expdir}
   
    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        echo "stage 2 : Decoding"
        nj=1 ## IA: was 16
        if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
            recog_model=model.last${n_average}.avg.best
            average_checkpoints.py --backend ${backend} \
                    --snapshots ${expdir}/results/snapshot.ep.* \
                    --out ${expdir}/results/${recog_model} \
                    --num ${n_average}
        fi
        pids=() # initialize pids
        #for rtask in ${recog_set}; do
        #(
            decode_dir=decode_${recog_set}_$(basename ${decode_config%.*})
            #feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

            # split data
            # splitjson.py --parts ${nj} ${feat_recog_dir}/data.json ## IA: old one
            # splitjson.py --parts ${nj} data/${recog_set}/data.json
            # IA: original was splitjson.py --parts ${nj} data/all_${rtask}/data.json

            #### use CPU for decoding (ngpu=0)
            ngpu=1

            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.log \
                asr_recog.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --debugmode ${debugmode} \
                --recog-json data/${recog_set}/data.json \
                --result-label ${expdir}/${decode_dir}/data.json \
                --model ${expdir}/results/${recog_model}

            score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
            # --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \ # IA: old one, for cmvn normalized feats
        # ) &
        pids+=($!) # store background pids
        #done
        i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

        echo "      Finished decoding reconstructed audio from en_${data_type}_reconstructed with the model from $expdir"
        stage=0
    fi
done

####################################### THE ORIGINAL run.sh script for the raw audio Librosa feats 22.02.2020 is below ##############


# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#     echo "stage -1: Data Download"
#     local/getdata.sh ${lang} ${voxforge}
# fi

# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#     ### Task dependent. You have to make data the following preparation part by yourself.
#     ### But you can utilize Kaldi recipes in most cases
#     echo "stage 0: Data Preparation"
#     selected=${voxforge}/${lang}/extracted
#     # Initial normalization of the data
#     local/voxforge_data_prep.sh ${selected} ${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020
#     local/voxforge_format_data.sh ${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020
# fi


# ############		IA: Wav -> mp3 -> wav conversion was done before running this script	########


# #feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
# #feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
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
#         # data/all_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 exp/make_fbank/train_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 ${fbankdir}

#     # steps/make_fbank.sh --cmd "$train_cmd" --nj 15 --write_utt2num_frames true \
#     #     data/all_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 exp/make_fbank/train_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 ${fbankdir}

#      ##################      IA: Generate the fbank features with LIBROSA; by default 80-dimensional fbanks on each frame
# # Synthax of make_fbank_librosa.sh: 
# # Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
# # e.g.: $0 data/train exp/make_fbank/train mfcc
# # Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data
#     fbankdir=fbank_librosa
#     make_fbank_librosa.sh --cmd "${train_cmd}" --nj 15 \
#         --fs ${fs} \
#         --fmax "${fmax}" \
#         --fmin "${fmin}" \
#         --n_fft ${n_fft} \
#         --n_shift ${n_shift} \
#         --win_length "${win_length}" \
#         --n_mels ${n_mels} \
#         data/all_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 \
#         data/all_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/log \
#         data/all_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/features
#     # IA original line was: --win_length "${win_length}" \
#     # I selected win_length as 512 according to Kaldi

#     utils/fix_data_dir.sh data/all_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020
#     ## IA: --nj 10 was original
#     ### IA make_fbank_pitch.sh Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
#     #  e.g.: $0 data/train
#     # Note: <log-dir> defaults to <data-dir>/log, and
#     #       <fbank-dir> defaults to <data-dir>/data

#     # remove utt having more than 2000 frames or less than 10 frames or
#     # remove utt having more than 200 characters or 0 characters
#     remove_longshortdata.sh data/all_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 data/all_trim_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020

#     # following split consider prompt duplication (but does not consider speaker overlap instead)
#     local/split_tr_dt_et.sh data/all_trim_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 \
#         data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 \
#         data/dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 \
#         data/et_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020

#     rm -r data/all_trim_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020

#     # compute global CMVN
#     #compute-cmvn-stats scp:data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/feats.scp data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/cmvn.ark

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
# #         data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/feats.scp data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
# #     dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
# #         data/dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/feats.scp data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
# #     for rtask in ${recog_set}; do
# #         feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
# #         dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
# #             data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
# #             ${feat_recog_dir}
# #     done
# fi

# dict=data/lang_1char_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/tr_${lang}_units.txt
# echo "dictionary: ${dict}"
# if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#     ### Task dependent. You have to check non-linguistic symbols used in the corpus.
#     echo "stage 2: Dictionary and Json Data Preparation"
#     mkdir -p data/lang_1char_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/
#     echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
#     text2token.py -s 1 -n 1 data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/text | cut -f 2- -d" " | tr " " "\n" \
#     | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
#     wc -l ${dict}

#     # make json labels
#     data2json.sh --lang ${lang} --feat data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/feats.scp \
#          data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 ${dict} > data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/data.json
#     data2json.sh --lang ${lang} --feat data/dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/feats.scp \
#          data/dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 ${dict} > data/dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/data.json
#     # for rtask in ${recog_set}; do
#     #     #feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
#     #     data2json.sh --feat ${feat_recog_dir}/feats.scp \
#     #         data/${rtask} ${dict} > ${feat_recog_dir}/data.json
#     # done

#     ##### IA: added
#     data2json.sh --lang ${lang} --feat data/et_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/feats.scp \
#          data/et_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 ${dict} > data/et_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/data.json
# fi

# if [ -z ${tag} ]; then
#     expname=${train_set}_${backend}_$(basename ${train_config%.*})
#     if ${do_delta}; then
#         expname=${expname}_delta
#     fi
# else
#     expname=${train_set}_${backend}_${tag}
# fi
# expdir=exp/${expname}
# mkdir -p ${expdir}

# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#     echo "stage 3: Network Training"
#     ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
#         asr_train.py \
#         --config ${train_config} \
#         --ngpu ${ngpu} \
#         --backend ${backend} \
#         --outdir ${expdir}/results \
#         --tensorboard-dir tensorboard/${expname} \
#         --debugmode ${debugmode} \
#         --dict ${dict} \
#         --debugdir ${expdir} \
#         --minibatches ${N} \
#         --verbose ${verbose} \
#         --resume ${resume} \
#         --train-json data/tr_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/data.json \
#         --valid-json data/dt_${lang}_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020/data.json
# fi

# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#     echo "stage 4: Decoding"
#     nj=16 ## IA: was 16
#     if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
#         recog_model=model.last${n_average}.avg.best
#         average_checkpoints.py --backend ${backend} \
# 			       --snapshots ${expdir}/results/snapshot.ep.* \
# 			       --out ${expdir}/results/${recog_model} \
# 			       --num ${n_average}
#     fi
#     pids=() # initialize pids
#     for rtask in ${recog_set}; do
#     (
#         decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
#         #feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

#         # split data
#         # splitjson.py --parts ${nj} ${feat_recog_dir}/data.json ## IA: old one
#         splitjson.py --parts ${nj} data/${rtask}/data.json

#         #### use CPU for decoding
#         ngpu=0

#         ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
#             asr_recog.py \
#             --config ${decode_config} \
#             --ngpu ${ngpu} \
#             --backend ${backend} \
#             --debugmode ${debugmode} \
#             --recog-json data/${rtask}/split${nj}utt/data.JOB.json \
#             --result-label ${expdir}/${decode_dir}/data.JOB.json \
#             --model ${expdir}/results/${recog_model}

#         score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
#         # --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \ # IA: old one, for cmvn normalized feats
#     ) &
#     pids+=($!) # store background pids
#     done
#     i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
#     [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
#     echo "Finished"
# fi

# if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
#     echo "stage 5: Creating the adversarial examples - 2 series of 8 jobs each (16 jobs in total)"
#     nj=16 ### IA: why 16 and not bigger, eg 32, like in R's script?! 
#     # IA: nj is only used in decoding loop below
#     if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
#         recog_model=model.last${n_average}.avg.best
#         average_checkpoints.py --backend ${backend} \
# 			       --snapshots ${expdir}/results/snapshot.ep.* \
# 			       --out ${expdir}/results/${recog_model} \
# 			       --num ${n_average}
#     fi
#     pids=() # initialize pids
#     #echo "test1"
#     for rtask in ${advEx_set}; do
#     (
#         #echo "test2"
#         decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
#         decode_dir_advex=decode_GPU_${rtask}_advEx_$(basename ${advex_config%.*})_${date_advex}
#         feat_recog_dir=data/${rtask}
#         ################### IA: Achtung at the csv_feats/ folder-name in the following line !!
#         mkdir -p ${expdir}/${decode_dir_advex}/csv_feats/

#         # split data  ##### IA: for faster decoding in parallel
#         #### IA: in adv ex creation phase, split data is commented because data was already splitted before
#        #  splitjson.py --parts ${nj} ${feat_recog_dir}/data.json ##### IA. data.json is already corrupted with duplicates :(

#         #### use CPU for decoding (###### IA: is GPU not good for decoding???)
#         ngpu=1
#         #echo "test3"
#         #### IA: here R did an outer for-loop to prevent the jobs from being run in parallel on the gpu because it took too much resources and got CUDA out of memory crash -> I dropped that loop 
#         #jobs=(1 9 17 25) #${jobs}
#         # #${decode_cmd} JOB=1 ${expdir}/${decode_dir_advex}/log/decode.JOB.log \
#         for start_job in 1 9; do 
#             #${decode_cmd} JOB=1:16 ${expdir}/${decode_dir_advex}/log/decode.JOB.log \
#             #### IA: make 4 series of 8 GPU AdvEx decoding jobs
#             #echo $i 
#             end_job=`expr $start_job + 7`
#             #echo $a
#             ${decode_cmd} JOB=${start_job}:${end_job} ${expdir}/${decode_dir_advex}/log/decode.JOB.log \
#                 asr_recog_advex.py \
#                 --config ${decode_config} \
#                 --ngpu ${ngpu} \
#                 --backend ${backend} \
#                 --debugmode ${debugmode} \
#                 --verbose ${verbose} \
#                 --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
#                 --result-label ${expdir}/${decode_dir_advex}/data.JOB.json \
#                 --model ${expdir}/results/${recog_model} \
#                 --path-target ${expdir}/${decode_dir}/data.JOB.json \
#                 --cuda-unit 0 \
#                 --advex-conf ${advex_config} ## IA: the decoding params for decoding the created adv ex
#         done
#         ### IA: expdir=exp/decoding_orig
#         #### IA: the JSON file with the original decoded transcriptions
#         ### JSON file from the exp dir, contains both the recognized text and ground truth transcription
#         ## IA: cuda-unit = 0, 1, 2, 3 ??
#         #### IA: asr_recog_advex.py is to be found in /home/iustina/espnet/espnet/bin/asr_recog_advex.py

#         # 1 3 5 6 7 10 11 12 (IA: these were the jobs not completed ! )

#         echo "Comparing now the adversarial example output to the original"
#         #### IA: actually comparison is done wrt the decoded transcripts, not the GND truth, or?
#         score_sclite.sh --wer true ${expdir}/${decode_dir_advex} ${dict}

#     ) &
#     pids+=($!) # store background pids
#     done
#     i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
#     [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

#     echo "Finished creation and decoding of adversarial examples"

#     ##### IA: syntax is: score_sclite_changed.sh --wer true $dir_folder_of_decoded_text $dictionary_folder
#     # IA: automatically reads the json file containing the decoded transcripts and originally decoded transcripts

#     # here the WER for the decoded text of the adv ex is computed by comparing with the original gnd truth text, not the decoded text from the non-adversarial input
#     # IA: really ??


# fi

# now=$(date)
# echo $now