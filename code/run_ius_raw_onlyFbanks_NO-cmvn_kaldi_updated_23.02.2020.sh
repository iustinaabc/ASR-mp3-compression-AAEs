#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false
cmvn=

train_config=conf/train_e4_subsamp12211_unit320_proj320_d1_300_mtlalpha0.3_epo20_ss-0.5_25.01.2020.yaml
decode_config=conf/decode_batch-sz10_beam-sz20_ctcw-0.3_12.03.2020.yaml
#decode_beam-sz10_ctcw-0.3_19.01.2020.yaml
fbank_config=conf/fbank_only_kaldi_23.02.2020_updated.conf

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

train_set=tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020
train_dev=dt_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020
recog_set="dt_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020 et_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/getdata.sh ${lang} ${voxforge}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    selected=${voxforge}/${lang}/extracted
    # Initial normalization of the data
    local/voxforge_data_prep.sh ${selected} ${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020
    local/voxforge_format_data.sh ${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020
fi



########################		Here convert the wav's to mp3s				###########################



#feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
#feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    #bankdir=fbank
    fbankdir=fbank_only_kaldi
    # Generate the fbank features; by default 80-dimensional ONLY_fbanks with pitch on each frame
    # steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 15 --write_utt2num_frames true \
        # data/all_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020 exp/make_fbank/train_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020 ${fbankdir}

    steps/make_fbank.sh --cmd "$train_cmd" --nj 15 --write_utt2num_frames true --fbank_config $fbank_config \
        data/all_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020 data/all_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/feat_extr_log ${fbankdir}
    utils/fix_data_dir.sh data/all_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020
    ## IA: --nj 10 was original
    ### IA make_fbank_pitch.sh Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
    #  e.g.: $0 data/train
    # Note: <log-dir> defaults to <data-dir>/log, and
    #       <fbank-dir> defaults to <data-dir>/data

    # remove utt having more than 2000 frames or less than 10 frames or
    # remove utt having more than 200 characters or 0 characters
    remove_longshortdata.sh data/all_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020 data/all_trim_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020

    # following split consider prompt duplication (but does not consider speaker overlap instead)
    local/split_tr_dt_et.sh data/all_trim_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020 data/tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020 data/dt_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020 data/et_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020
    rm -r data/all_trim_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020

    # compute global CMVN
    #compute-cmvn-stats scp:data/tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/feats.scp data/tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/cmvn.ark

    # dump features for training
#     if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
#     utils/create_split_dir.pl \
#         /export/b{14,15,16,17}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_set}/delta${do_delta}/storage \
#         ${feat_tr_dir}/storage
#     fi
#     if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
#     utils/create_split_dir.pl \
#         /export/b{14,15,16,17}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_dev}/delta${do_delta}/storage \
#         ${feat_dt_dir}/storage
#     fi
#     dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
#         data/tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/feats.scp data/tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
#     dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
#         data/dt_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/feats.scp data/tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
#     for rtask in ${recog_set}; do
#         feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
#         dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
#             data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
#             ${feat_recog_dir}
#     done
 fi

dict=data/lang_1char_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/tr_${lang}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --lang ${lang} --feat data/tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/feats.scp \
         data/tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020 ${dict} > data/tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/data.json
    data2json.sh --lang ${lang} --feat data/dt_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/feats.scp \
         data/dt_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020 ${dict} > data/dt_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/data.json
    # for rtask in ${recog_set}; do
    #     #feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
    #     data2json.sh --feat ${feat_recog_dir}/feats.scp \
    #         data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    # done

    ##### IA: added
    data2json.sh --lang ${lang} --feat data/et_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/feats.scp \
         data/et_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020 ${dict} > data/et_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/data.json
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json data/tr_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/data.json \
        --valid-json data/dt_${lang}_raw_ONLY_fbanks_NO-cmvn_Kaldi_updated_23.02.2020/data.json
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Decoding"
    nj=1 ## IA: when decoding with GPU, max # of jobs is 1, because loading the model already takes half of the GPU !!!! if njobs is bigger than 1, the program will crash
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
			       --snapshots ${expdir}/results/snapshot.ep.* \
			       --out ${expdir}/results/${recog_model} \
			       --num ${n_average}
    fi
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        #feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        # splitjson.py --parts ${nj} ${feat_recog_dir}/data.json ## IA: old one
        #splitjson.py --parts ${nj} data/${rtask}/data.json

        #### use CPU for decoding
        ngpu=1

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --recog-json data/${rtask}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
        # --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \ # IA: old one, for cmvn normalized feats
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi