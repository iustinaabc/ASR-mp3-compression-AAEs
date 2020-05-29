#!/bin/bash

### IA: this script is intented to augment the original test datasets with different types of noises (white, pink, brown, babble) at various SNRs, then mp3 compress them (at 24kbps for raw, 128 and 64 kbps original data and at 16 kbps for the 24kbps original data), then decode them with the models trained on the corresponding set (raw, 128, 64 and 24 kbps original data)

# 1. mp3-compress them

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

#kbps=16 # IA: see the cmpr_bitrate variable below
date=03.03.2020

fs=16000      # sampling frequency
fmax=7800     # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=512    # number of fft points
n_shift=160   # number of shift points ### IA: was originally 256
#### IA: overlap of 10 ms is the default in Kaldi, which for us means 10*1e-3*16000 = 160 (points)
win_length=512 # window length
### IA: in librosa.core.stft, If unspecified, defaults to win_length = n_fft
## https://librosa.github.io/librosa/generated/librosa.core.stft.html?highlight=stft#librosa.core.stft
window=hann

compress=true
normalize=16  # The bit-depth of the input wav files
filetype=mat
lang=en

# feature configuration
do_delta=false

decode_config=conf/decode_batch-sz100_beam-sz20_ctcw-0.3_12.03.2020.yaml
#decode_beam-sz10_ctcw-0.3_19.01.2020.yaml

data_basename=compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn
#compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn
#compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn
#raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020
#compr_128kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn
exp_name=tr_en_compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_24.02.2020
#tr_en_compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_26.02.2020
# tr_en_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_23.02.2020
# tr_en_compr_128kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo25_ss0.5_28.02.2020

noises="whitenoise pinknoise brownnoise babblenoise"
snrs="30 10 5 0 -5 -10"
cmpr_bitrate=16
# !!!!!!!! Achtung: when compressing at bitrates below 16kbps (exclusive 16kbps), use the -s 16 in second lame cmd from the pipe, otherwise the compressed samples will have 8k sampling rate instead of 16k

# IA: all the variables before the following line (with the parse_options.sh cmd) can be overwritten by specifying command line arguments when running this script

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

orig_data_dir=data/et_en_$data_basename
expdir="exp/$exp_name"
dict=data/lang_1char_$data_basename/tr_${lang}_units.txt

# We want to augment data_basename with noises of different types and SNR - we thus create diff noises and noise data dirs 

### IA: check if noise dir exists. If yes, we dont compute it again!
if [ ! -d noise/ ]; then
    mkdir -p noise/ # the dir where actual .sph noise is used 
    for ns in "whitenoise" "pinknoise" "brownnoise"; do
        DURATION=5
        #touch noise/${ns}.sph
        sox -n -r 16k -b 16 noise/${ns}.sph synth ${DURATION} ${ns}
        mkdir -p data/noise/${ns}
        echo "${ns} /home/iustina/espnet/tools/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p $(pwd)/noise/${ns}.sph |" | tee data/noise/${ns}/wav.scp
        echo "${ns} ${DURATION}" > data/noise/${ns}/reco2dur
    done
else 
    echo -e "  Noise files already exist! They are not recomputed. "
fi

echo -e " \n   exp folder is $expdir"
# Noise augmentation + compression + decoding loop 
for ns in $noises; do
# 
    for snr in $snrs ; do
        augm_data_dir=${orig_data_dir}/augmented_${ns}_${snr}SNR_compr_${cmpr_bitrate}kbps

        if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
            echo -e " \n   Augmenting et_en_${data_basename} with $ns at $snr SNR" 
            utils/data/get_reco2dur.sh ${orig_data_dir}
            # create wav.scp
            #Usage: augment_data_dir.py [options...] <in-data-dir> <out-data-dir> 
            python3 steps/data/augment_data_dir.py --bg-noise-dir data/noise/${ns} --bg-snrs ${snr} ${orig_data_dir} ${augm_data_dir}

            # IA: append the mp3 24/16/10 kbps compression pipe cmd with lame at the end of each line from the created wav.scp
            # there might be some error caused by the durations, because compression can increase/decrease duration
            sed -e 's/$/ lame --silent -b 16 - -| lame --silent --decode --mp3input - - |/' -i ${augm_data_dir}/wav.scp 
            ###                 IA: careful when selecting compression bitrate!!!
            # !!!!!!!! Also, when compressing at bitrates below 16kbps, use the -s 16 in second lame cmd from the pipe, otherwise the compressed samples will have 8k sampling rate instead of 16k

            # copy the other needed files (eg. spk2utt and utt2spk, text) for feats computation
            cp -v ${orig_data_dir}/{utt2spk,spk2utt,text} ${augm_data_dir}/
        fi
        
        #recog_set=$advEx_wav_dir

        if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
            echo -e "\n       stage 2: extract new features from $cmpr_bitrate kbps mp3-COMPRessed ${augm_data_dir} \n make_fbank_librosa.sh -> (new_augm_Data_)feats.scp"
            #### IA: for this, I need to have the proper wav.scp utt2spkr and all the other files in the same folder with new_compressed_AdvEx_feats.scp
            ### Obs. the new_compressed_AdvEx_feats.scp does not need to have the original mappings btw the utterance ids and wav paths

            # Synthax of make_fbank_librosa.sh: 
            # Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]

            make_fbank_librosa.sh --cmd "${train_cmd}" --nj 15 \
                --fs ${fs} \
                --fmax "${fmax}" \
                --fmin "${fmin}" \
                --n_fft ${n_fft} \
                --n_shift ${n_shift} \
                --win_length "${win_length}" \
                --n_mels ${n_mels} \
                --window ${window} \
                $augm_data_dir \
                $augm_data_dir/feature_extr_log \
                $augm_data_dir/features_librosa
            # $1: folder where the file wav.scp of compressed AdvEx is (eg. ${expdir}/advex_compr_${kbps}kbps ) 
        fi

        if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
            echo -e "\n      stage 3: JSON file creation from the new feats.scp of COMPRessed ${augm_data_dir}"
            ## for each wav in $advex_audio_folder (./exp/${expdir}/advEx_audio_from_csv)
            #       sox conversion cmd
            # end

            data2json.sh --lang ${lang} --feat ${augm_data_dir}/feats.scp \
                ${augm_data_dir} ${dict} > ${augm_data_dir}/data.json
        fi


        if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
            echo -e "\n     stage 4: Decoding the data from the new JSON file (corresponding to ${augm_data_dir})"
            nj=1 ## IA: was 16
            # if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
            #     recog_model=model.last${n_average}.avg.best
            #     average_checkpoints.py --backend ${backend} \
            #             --snapshots ${expdir}/results/snapshot.ep.* \
            #             --out ${expdir}/results/${recog_model} \
            #             --num ${n_average}
            # fi
            # pids=() # initialize pids

            decode_dir=decode_et_en_${data_basename}_augm_${ns}_${snr}SNR_compr_${cmpr_bitrate}kbps_$(basename ${decode_config%.*})
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
                --recog-json ${augm_data_dir}/data.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model}

            ### IA: for CPU decoding, select 16 jobs --recog-json ${augm_data_dir}/split${nj}utt/data.JOB.json \
            ### For GPU decoding, select only 1 !!! job and --recog-json ${augm_data_dir}/data.json \

            score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
            # --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \ # IA: old one, for cmvn normalized feats
            # pids+=($!) # store background pids



            # i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
            # [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
            echo -e "      Finished decoding for ${augm_data_dir}"
        fi
        stage=1
    done
done # end of advEx decoding loop for all the advEx sets