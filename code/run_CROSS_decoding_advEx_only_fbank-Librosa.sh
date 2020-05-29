#!/bin/bash

### IA: this script is intented to take the audio AdvEx files (previously converted from .csv with csv2wav_loop_ius.py) and:
# 1. mp3-compress them
# 2. extract new features from the compressed AdvEx (make_fbank_librosa.sh) -> new_feats.scp
# 3. build the json file from the new_feats.scp (data2json.sh) -> advex_compressed_data.json
# 4. decode the advex_compressed_data.json
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

model_basenames="raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 compr_128kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn"
# compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn
#"raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 compr_128kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn"

# IA: model_basenames represent the type of data on which the model with which we choose to decode was originally trained

advEx_root_basenames="raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020 compr_128kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn"

# data type from which orig and inverted advEx were created 

advEx_type_basenames="advEx_audio_from_csv_orig_espnet_method_noNorm advEx_compr_128kbps_FULLscale_espnet_noNorm  advEx_compr_64kbps_FULLscale_espnet_noNorm advEx_compr_24kbps_FULLscale_espnet_noNorm advEx_compr_16kbps_FULLscale_espnet_noNorm"


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


#dict=data/lang_1char_$data_basenames/tr_${lang}_units.txt

# We want to augment data_basename with noises of different types and SNR - we thus create diff noises and noise data dirs 
echo -e "       Decoding config is $decode_config"
# Decoding loop
for model in $model_basenames; do # model that will do the decoding
    for advEx_root in $advEx_root_basenames; do  # data from which orig and inverted advEx were created 
        for advEx_type in $advEx_type_basenames; do
            if [ $model == $advEx_root ]; then
                echo -e "\n     REPEAT decoding $advEx_type computed from data/et_en_$advEx_root with model trained on $model"
            else 
                echo -e "\n     CROSS-test: Decoding $advEx_type computed from data/et_en_$advEx_root with model trained on $model"
            fi         


            advEx_data_dir=data/et_en_${advEx_root}_${advEx_type} # dir containing the json file of the desired AdvEx to be decoded 
            if [ ! -e $advEx_data_dir ]; then
                echo -e "\n    CANNOT decode $advEx_type computed from data/et_en_$advEx_root with model trained on $model because the directory $advEx_data_dir does NOT exist! "
                continue
            fi
            
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

            # if [ $model==$data ]; then
            #     decode_dir=decode_REPEAT_et_en_${data}_$(basename ${decode_config%.*})
            # else 
            #     decode_dir=decode_CROSStest_et_en_${data}_$(basename ${decode_config%.*})
            # fi  

            decode_dir=decode_CROSStest_advEx_et_en_${advEx_root}_${advEx_type}_$(basename ${decode_config%.*})
            if [ -e $expdir/${decode_dir}/result.txt ]; then # check if we already decoded this before -> skip it !
                echo -e "\n    SKIP: $advEx_type computed from data/et_en_$advEx_root was already decoded with model trained on $model ! \n ${decode_dir}/result.txt already exists!"
                continue
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
                --recog-json ${advEx_data_dir}/data.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model}

            score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
            # --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \ # IA: old one, for cmvn normalized feats
            #           IA: for CPU decoding, use 16 jobs and
            # --recog-json ${data_dir}/split${nj}utt/data.JOB.json \
            #           IA: for GPU decoding, use 1 !!! job and
            # --recog-json ${data_dir}/data.json \

            # pids+=($!) # store background pids
            
            # i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
            # [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

            if [ $model == $advEx_root ]; then
                echo -e "      Finished REPEATED decoding $advEx_type computed from data/et_en_$advEx_root with model trained on $model \n "
            else 
                echo -e "      Finished CROSS-test: Decoding $advEx_type computed from data/et_en_$advEx_root with model trained on $model \n "
            fi 
        done
    done
done # end of advEx decoding loop for all the advEx sets