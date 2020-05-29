#!/bin/bash

### 13.03.2020
# IA: this script is intented to run CROSS decoding of AdvEx
### More specifically, we take the original inverted advEx from a certain type of input data (raw, 128, 64, 24 kbps compressed), as well as compressed advEx and decode it with models trained on other types of data

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

backend=pytorch
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
stage=1
stop_stage=100
debugmode=1
verbose=1      # verbose option
resume=       
recog_model=model.acc.best

#kbps=24

compress=true
normalize=16  # The bit-depth of the input wav files
filetype=mat
lang=en

# feature configuration
do_delta=false

decode_config=conf/decode_batch-sz100_beam-sz20_ctcw-0.3_12.03.2020.yaml
#decode_beam-sz10_ctcw-0.3_19.01.2020.yaml


# IA: in exp_name resides the already trained model -> so exp_name will denote the type of data with which the decoding model was trained
# exp_name=tr_en_compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_24.02.2020
#tr_en_compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_26.02.2020
# tr_en_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_23.02.2020
# tr_en_compr_128kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo25_ss0.5_28.02.2020

# noises="whitenoise pinknoise brownnoise babblenoise"
# snrs="30 10 5 0 -5 -10"

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

data_basenames="raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 compr_128kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn"
# 


model_basenames=$data_basenames

#dict=data/lang_1char_$data_basenames/tr_${lang}_units.txt

# We want to augment data_basename with noises of different types and SNR - we thus create diff noises and noise data dirs 

# Decoding loop
for model in $model_basenames; do
    for data in $data_basenames; do
        echo -e "       Decoding config is $decode_config"

        if [ $model==$data ]; then
            echo -e "\n    REPEAT decoding data/et_en_$data with model trained on $model"
        else 
            echo -e "\n    CROSS-test: Decoding data/et_en_$data with model trained on $model"
        fi   

        data_dir=data/et_en_$data
        dict=data/lang_1char_$model/tr_${lang}_units.txt
        n_epochs="20"

        if [ $model == raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 ]; then
            date="23.02.2020"
            
        elif [ $model == compr_128kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn ]; then
            date="28.02.2020"
            n_epochs="25"

        elif [ $model == compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn ]; then
            date="24.02.2020"

        elif [ $model == compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn ]; then
            date="26.02.2020"
        fi

        # if [ $marks -ge 80 ]
        # then
        #     echo "Very Good"
        
        # elif [ $marks -ge 50 ]
        # then
        #     echo "Good"
        
        # elif [ $marks -ge 33 ]
        # then
        #     echo "Just Satisfactory"
        # else
        #     echo "Not OK"
        # fi
                
        exp_name=tr_en_${model}_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo${n_epochs}_ss0.5_$date

        expdir="exp/$exp_name"   

        echo -e "       The dir from which model is taken is $expdir"
        nj=1 ## IA: was 16
        # if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        #     recog_model=model.last${n_average}.avg.best
        #     average_checkpoints.py --backend ${backend} \
        #             --snapshots ${expdir}/results/snapshot.ep.* \
        #             --out ${expdir}/results/${recog_model} \
        #             --num ${n_average}
        # fi
        # pids=() # initialize pids

        if [ $model==$data ]; then
            decode_dir=decode_REPEAT_et_en_${data}_$(basename ${decode_config%.*})
        else 
            decode_dir=decode_CROSStest_et_en_${data}_$(basename ${decode_config%.*})
        fi   
        
        #feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        # splitjson.py --parts ${nj} ${feat_recog_dir}/data.json ## IA: old one
        #splitjson.py --parts ${nj} ${augm_data_dir}/data.json
        

        #### use CPU for decoding
        #ngpu=0

        # IA: Original 
        # ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        #     asr_recog.py \
        #     --config ${decode_config} \
        #     --ngpu ${ngpu} \
        #     --backend ${backend} \
        #     --debugmode ${debugmode} \
        #     --recog-json data/${rtask}/split${nj}utt/data.JOB.json \
        #     --result-label ${expdir}/${decode_dir}/data.JOB.json \
        #     --model ${expdir}/results/${recog_model}

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --recog-json ${data_dir}/data.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
        # --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \ # IA: old one, for cmvn normalized feats
        #           IA: for CPU decoding, use 16 jobs and
        # --recog-json ${data_dir}/split${nj}utt/data.JOB.json \
        #   IA: for GPU decoding, use 1 !!! job and
        # --recog-json ${data_dir}/data.json \

        # pids+=($!) # store background pids
        
        # i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        # [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
        echo -e "      Finished decoding data/et_en_$data with model trained on $model}"
    done
done # end of advEx decoding loop for all the advEx sets