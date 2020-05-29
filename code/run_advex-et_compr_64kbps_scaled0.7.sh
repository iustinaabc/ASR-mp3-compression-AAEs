#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

##### IA: no 'tag' required for creating AdvEx, because I will output them to the exp folder of the respective trained model that I use; I should just be careful to modify expdir (name of exp directory to be used) accordingly 

# now=$(date)
# echo $now

. ./path.sh #|| exit 1; #IA: '|| exit 1;' is commented in Ricardo's run_advex_db.sh
# . ./cmd.sh #|| exit 1; # IA: this was run in R's run script way below and without '|| exit 1;'

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option ################## IA: I modified this from 0 to 2 for debugging mode
resume=        # Resume the training from snapshot


# feature configuration
do_delta=false
cmvn=

###### IA: select_appropriate training config files - not relevant now for adv ex
train_config=conf/train_e4_subsamp12211_unit320_proj320_d1_300_mtlalpha0.3_epo25_ss-0.5_19.01.2020.yaml
decode_config_advex=conf/decode_beam-sz10_ctcw-0.3_batchsz35_30.01.2020.yaml #ORIGINAL decoding conf, used after training the net: conf/decode_beam-sz10_ctcw-0.3_19.01.2020.yaml; I change the batchsize from 0 to 35 to speed up decoding of AdvEx
decode_config_orig=conf/decode_beam-sz10_ctcw-0.3_19.01.2020.yaml
advex_config=conf/advex_conf_bsz10_MovWin_eps0.3_winsz4_str2.yaml ################ IA: decoding config for the adversarial examples

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10 # IA: ????? in selecting the recog_model in decoding stage below

#decode_advex=adv_ex
#decode_model=decode
#api=v1


# data
voxforge=downloads # original data directory to be stored
lang=en # de, en, es, fr, it, nl, pt, ru

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# make shellcheck happy (added by IA)
train_cmd=
decode_cmd=

. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=tr_${lang}_compr_64kbps_scaled0.7
train_dev=dt_${lang}_compr_64kbps_scaled0.7
recog_set="et_${lang}_compr_64kbps_scaled0.7" # " dt_${lang}_compr_64kbps_scaled0.7"
#### IA: above line chooses only dev set 

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/getdata.sh ${lang} ${voxforge}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"


    ########## IA: selection of wavs with different compression bitrates !!! (here 64 kbps)

    selected=${voxforge}/${lang}_compr_mp3_64kbps_scaled0.7/extracted


    # Initial normalization of the data
    local/voxforge_data_prep.sh ${selected} ${lang}_compr_64kbps_scaled0.7
    local/voxforge_format_data.sh ${lang}_compr_64kbps_scaled0.7
fi



########################		Here convert the wav's to mp3s				###########################


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    ##### IA: probably this section is redundant for the adv_ex, we'll already have computed the features before training the model (which we further use to create adv ex)

    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
        data/all_${lang}_compr_64kbps_scaled0.7 exp/make_fbank/train_${lang}_compr_64kbps_scaled0.7 ${fbankdir}
    utils/fix_data_dir.sh data/all_${lang}_compr_64kbps_scaled0.7

    # remove utt having more than 2000 frames or less than 10 frames or
    # remove utt having more than 200 characters or 0 characters
    remove_longshortdata.sh data/all_${lang}_compr_64kbps_scaled0.7 data/all_trim_${lang}_compr_64kbps_scaled0.7

    # following split consider prompt duplication (but does not consider speaker overlap instead)
    local/split_tr_dt_et.sh data/all_trim_${lang}_compr_64kbps_scaled0.7 data/tr_${lang}_compr_64kbps_scaled0.7 data/dt_${lang}_compr_64kbps_scaled0.7 data/et_${lang}_compr_64kbps_scaled0.7
    rm -r data/all_trim_${lang}_compr_64kbps_scaled0.7

    # compute global CMVN
    compute-cmvn-stats scp:data/tr_${lang}_compr_64kbps_scaled0.7/feats.scp data/tr_${lang}_compr_64kbps_scaled0.7/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/voxforge/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
        data/tr_${lang}_compr_64kbps_scaled0.7/feats.scp data/tr_${lang}_compr_64kbps_scaled0.7/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        data/dt_${lang}_compr_64kbps_scaled0.7/feats.scp data/tr_${lang}_compr_64kbps_scaled0.7/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char_compr_64kbps_scaled0.7/tr_${lang}_units.txt
echo "dictionary: ${dict}" ################## IA: probably redundant for the adv_ex, we already computed the dic before training the model (which we further use to create adv ex)
#### IA: so we don't need to compute it again, nor de we use a bpe model

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char_compr_64kbps_scaled0.7/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/tr_${lang}_compr_64kbps_scaled0.7/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --lang ${lang} --feat ${feat_tr_dir}/feats.scp \
         data/tr_${lang}_compr_64kbps_scaled0.7 ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --lang ${lang} --feat ${feat_dt_dir}/feats.scp \
         data/dt_${lang}_compr_64kbps_scaled0.7 ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

##### IA: original expdir computed below
# if [ -z ${tag} ]; then
#     expname=${train_set}_$(basename ${train_config%.*})
#     if ${do_delta}; then
#         expname=${expname}_delta
#     fi
# else
#     expname=${train_set}_${tag}
# fi
# expdir=exp/${expname}
# mkdir -p ${expdir}

expdir=exp/tr_en_compr_64kbps_scaled0.7_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo25_ss-0.5_19.01.2020

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
#         --train-json ${feat_tr_dir}/data.json \
#         --valid-json ${feat_dt_dir}/data.json
# fi

######## IA: following section is commented because the decoding of the original dev and test sets was already performed after the training stage !!

# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#     echo "stage 3: Decoding of original utterances"
#     nj=16 ### IA: why not bigger, eg 32 ?! only used in decoding loop
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
#         feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

#         # split data  ##### IA: for faster decoding in parallel???
#         splitjson.py --parts ${nj} ${feat_recog_dir}/data.json ##### IA. data.json is already corrupted with duplicates :(

#         #### use CPU for decoding (###### IA: is GPU not good for decoding???)
#         ngpu=0

#         ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
#             asr_recog.py \
#             --config ${decode_config} \
#             --ngpu ${ngpu} \
#             --backend ${backend} \
#             --debugmode ${debugmode} \
#             --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
#             --result-label ${expdir}/${decode_dir}/data.JOB.json \
#             --model ${expdir}/results/${recog_model}

#         score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

#     ) &
#     pids+=($!) # store background pids
#     done
#     i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
#     [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
#     echo "Finished decoding of original utterances"
# fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Creating the adversarial examples"
    nj=16 ### IA: why 16 and not bigger, eg 32, like in R's script?! 
    # IA: nj is only used in decoding loop below
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
			       --snapshots ${expdir}/results/snapshot.ep.* \
			       --out ${expdir}/results/${recog_model} \
			       --num ${n_average}
    fi
    pids=() # initialize pids
    #echo "test1"
    for rtask in ${recog_set}; do
    (
        #echo "test2"
        decode_dir=decode_${rtask}_$(basename ${decode_config_orig%.*})
        decode_dir_advex=decode_GPU_${rtask}_advEx_$(basename ${advex_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        mkdir -p ${expdir}/${decode_dir_advex}/csv_feats/

        # split data  ##### IA: for faster decoding in parallel

        #### IA: in adv ex creation phase, split data is commented because data was already splitted before
       #  splitjson.py --parts ${nj} ${feat_recog_dir}/data.json ##### IA. data.json is already corrupted with duplicates :(

        #### use CPU for decoding (###### IA: is GPU not good for decoding???)
        ngpu=1
        # nj=1
        # =23
        #echo "test3"

        #### IA: here R did an outer for-loop to prevent the jobs from being run in parallel on the gpu because it took too much resources and got CUDA out of memory crash -> anyhow I dropped that loop
        for index in $(seq 1 8); 
        do
            ${decode_cmd} JOB=${index}:${index} ${expdir}/${decode_dir_advex}/log/decode.JOB.log \
                asr_recog_advex.py \
                --config ${decode_config_advex} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --debugmode ${debugmode} \
                --verbose ${verbose} \
                --recog-json ${feat_recog_dir}/split16utt/data.JOB.json \
                --result-label ${expdir}/${decode_dir_advex}/data.JOB.json \
                --model ${expdir}/results/${recog_model} \
                --path-target ${expdir}/${decode_dir}/data.JOB.json \
                --cuda-unit 0 \
                --advex-conf ${advex_config} ## IA: the decoding params for decoding the created adv ex
            # IA: --recog-json is the json file from dump folder, contains mainly data about the input - features + ground truth transcription
            #IA: expdir=exp/decoding_orig in R'S original code
            ##### IA: --result-label is the JSON file where to write the AdvEx DECODED transcriptions
            #### IA: --path-target is JSON file from the exp dir, contains both the recognized text and ground truth transcription
            ### IA: cuda-unit = 0, 1, 2, 3 ?? (R said it doesn't matter; same process runs irrespective of the # chosen here)
            ##### IA: asr_recog_advex.py is to be found in /home/iustina/espnet/espnet/bin/asr_recog_advex.py
        done

        echo "Comparing now the adversarial example output to the original"
        #### IA: actually comparison is done wrt the decoded transcripts, not the GND truth, or?
        score_sclite.sh --wer true ${expdir}/${decode_dir_advex} ${dict}
        ##### IA: syntax is: score_sclite_changed.sh --wer true $dir_folder_of_decoded_text $dictionary_folder
        # IA: automatically reads the json file containing the decoded transcripts and originally decoded transcripts

        # here the WER for the decoded text of the adv ex is computed by comparing with the original gnd truth text, not the decoded text from the non-adversarial input
        # IA: really????

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    echo "Finished creation and decoding of adversarial examples"
fi

# now=$(date)
# echo $now