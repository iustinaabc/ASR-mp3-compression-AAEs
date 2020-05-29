#!/bin/bash

### IA: this script is intented to take the audio AdvEx files (previously converted from .csv with csv2wav_loop_ius.py) and:
# 1. mp3-compress them
# 2. extract new features from the compressed AdvEx (make_fbank_librosa.sh) -> new_feats.scp
# 3. build the json file from the new_feats.scp (data2json.sh) -> advex_compressed_data.json
# 4. decode the advex_compressed_data.json
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

backend=pytorch
#ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
#stage=2
#stop_stage=100
debugmode=1
verbose=1      # verbose option
resume=       
recog_model=model.acc.best

#kbps=24
date=01.03.2020

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
#conf/decode_beam-sz10_ctcw-0.3_19.01.2020.yaml
#conf/decode_beam-sz10_ctcw-0.5_06.03.2020.yaml
#conf/decode_beam-sz10_ctcw-0.1_06.03.2020.yaml
#conf/decode_beam-sz10_ctcw-0.3_19.01.2020.yaml # ORIGINAL

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

advEx_wav_dir=advEx_compr_16kbps_FULLscale_espnet_noNorm
# "advEx_audio_from_csv_orig_espnet_method_noNorm advEx_compr_128kbps_FULLscale_espnet_noNorm advEx_compr_64kbps_FULLscale_espnet_noNorm advEx_compr_24kbps_FULLscale_espnet_noNorm advEx_compr_16kbps_FULLscale_espnet_noNorm"

#advEx_audio_from_csv_orig_espnet_method_noNorm
#advEx_compr_64kbps_scaled_0.7_espnet_max-norm
#advEx_compr_10kbps_FULLscale_espnet_noNorm 
#advEx_audio_from_csv_orig_espnet_method_noNorm 
#advex_compr_24kbps_FULLscale_espnet_noNorm
data_basename=compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn
#raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020
#compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn
#compr_128kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn
#compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn
orig_data_dir=data/et_en_$data_basename
exp_name=tr_en_compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_24.02.2020
#tr_en_raw_ONLY_fbanks-Librosa_NO-cmvn_22.02.2020_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_23.02.2020
#tr_en_compr_24kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_26.02.2020
#tr_en_compr_128kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo25_ss0.5_28.02.2020
#tr_en_compr_64kbps_scaled0.7_ONLY_fbanks-Librosa_NO-cmvn_pytorch_e4_12211_u320_proj320_d1_300_mtlalpha0.3_epo20_ss0.5_24.02.2020
expdir="exp/$exp_name"
dict=data/lang_1char_$data_basename/tr_${lang}_units.txt

# AdvEx decoding loop 
for rtask in ${advEx_wav_dir}; 
do
    echo -e " \n    Processing $rtask"

    ### For the moment, assume we already have the compressed AdvEx and we'll just read them from their folder
    # But before feature extraction, we need to somehow bypass voxforge data preparation because for AdvEx we don't have the same folder structure as for the original data (ie. downloads/en/extracted/speaker_folder etc). So we aim to manually create the appropriate wav.scp file (with utterance names and AdvEx paths). As for the rest of files (utt2spk, spk2utt, text), we simply copy them from the orig_data_dir.
    
        advEx_data_dir=${orig_data_dir}_$rtask
        mkdir -p $advEx_data_dir
    
    recog_set=$advEx_wav_dir

    #if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        echo -e "\n      stage 1: Data dir preparation for $rtask"
        
        # First, get the name of all the utterances in the test set (extract them from the original wav.scp in orig_data_dir - they should already be alphabetically sorted)
        echo "      Create file with utterance ids of the curent test set"
        awk '{print $1}' $orig_data_dir/wav.scp > $orig_data_dir/utt_names_ius.txt
        utt_names_file=$orig_data_dir/utt_names_ius.txt

        echo -e "      Create the new wav.scp for the target compressed advEx ($rtask)"
        # Create the new wav.scp for our advEx
        find $expdir/$rtask -type f -iname '*wav' -print | sort | paste -d ' ' $utt_names_file - > $advEx_data_dir/wav.scp
        # above pipe searches for the paths (relative to the current folder) of the desired advEx, then sorts the list based on the wav filenames (so that the advEx are in the correct alphabetical order and match the order of the utterance names). Then, the paths are concatenated to the file containing the utterance names, then the new wav.scp is written in the corresponding advEx data directory

        echo -e "      Copy the other files (utt2spk, spk2utt, text and utt2num_frames) from the original test set data dir ($orig_data_dir)"
        # Then copy the utt2spk, spk2utt, text and utt2num_frames (?) in the new AdvEx data folder 
        # Sytax copy $origin_folder $destination_folder
        cp $orig_data_dir/{utt2spk,spk2utt,text,utt2num_frames} $advEx_data_dir

    #fi


    #if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        echo -e "\n       stage 2: extract new features from $rtask \n make_fbank_librosa.sh -> (new_compr_AdvEx_)feats.scp in $advEx_data_dir"
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
            $advEx_data_dir \
            $advEx_data_dir/feature_extr_log \
            $advEx_data_dir/features_librosa
        # $1: folder where the file wav.scp of compressed AdvEx is (eg. ${expdir}/advex_compr_${kbps}kbps ) 
    # fi

    #if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        echo -e "\n      stage 3: JSON file creation from the new feats.scp of $rtask"
        ## for each wav in $advex_audio_folder (./exp/${expdir}/advEx_audio_from_csv)
        #       sox conversion cmd
        # end

        data2json.sh --lang ${lang} --feat $advEx_data_dir/feats.scp \
            $advEx_data_dir ${dict} > $advEx_data_dir/data.json
    #fi


    # #if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    #     echo -e "\n     stage 4: Decoding the data from the new json file (corresponding to $rtask)"
    #     nj=1 ## IA: was 16
    #     # if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
    #     #     recog_model=model.last${n_average}.avg.best
    #     #     average_checkpoints.py --backend ${backend} \
    #     #             --snapshots ${expdir}/results/snapshot.ep.* \
    #     #             --out ${expdir}/results/${recog_model} \
    #     #             --num ${n_average}
    #     # fi
    #     # pids=() # initialize pids

    #     decode_dir=decode_et_${rtask}_$(basename ${decode_config%.*})
    #     #feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

    #     # split data
    #     # splitjson.py --parts ${nj} ${feat_recog_dir}/data.json ## IA: old one
    #     #splitjson.py --parts ${nj} $advEx_data_dir/data.json
        

    #     #### use CPU for decoding
    #     ngpu=1

    #     # IA: Original 
    #     # ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
    #     #     asr_recog.py \
    #     #     --config ${decode_config} \
    #     #     --ngpu ${ngpu} \
    #     #     --backend ${backend} \
    #     #     --debugmode ${debugmode} \
    #     #     --recog-json data/${rtask}/split${nj}utt/data.JOB.json \
    #     #     --result-label ${expdir}/${decode_dir}/data.JOB.json \
    #     #     --model ${expdir}/results/${recog_model}

    #     ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
    #         asr_recog.py \
    #         --config ${decode_config} \
    #         --ngpu ${ngpu} \
    #         --backend ${backend} \
    #         --debugmode ${debugmode} \
    #         --recog-json ${advEx_data_dir}/data.json \
    #         --result-label ${expdir}/${decode_dir}/data.JOB.json \
    #         --model ${expdir}/results/${recog_model}
        
    #     # IA: For CPU decoding, nj=16, ngpu=0 and --recog-json ${advEx_data_dir}/split${nj}utt/data.JOB.json \
    #     # IA: For GCPU decoding, nj=1, ngpu=1 and --recog-json ${advEx_data_dir}/data.json \

    #     score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}
    #     # --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \ # IA: old one, for cmvn normalized feats
    #     # pids+=($!) # store background pids
        
    #     # i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    #     # [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    #     echo -e "      Finished decoding for $rtask"
    # #fi
done # end of advEx decoding loop for all the advEx sets