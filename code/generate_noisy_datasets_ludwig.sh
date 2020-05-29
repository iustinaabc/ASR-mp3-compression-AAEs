#!/bin/env bash

# generate noise for the tedlium 2 dataset in kaldi
. path.sh
. cmd.sh
$KALDI_ROOT/egs/wsj/s5/utils/data/get_reco2dur.sh data/test
$KALDI_ROOT/egs/wsj/s5/utils/data/get_reco2dur.sh data/dev
mkdir -p db/noise
for ns in "noise" "whitenoise" "pinknoise" "brownnoise"; do
    DURATION=5
    sox -n -r 16k -b 16 db/noise/${ns}.sph synth ${DURATION} ${ns}
    mkdir -p data/${ns}
    echo "${ns} sph2pipe -f wav -p $(pwd)/db/noise/${ns}.sph |" | tee data/${ns}/wav.scp
    echo "${ns} ${DURATION}" > data/${ns}/reco2dur
    for snr in "50"  "40"  "30"  "25"  "20"  "15"  "10"  "5"  "0"  "-5"  "-10" ; do
        echo "SNR: " ${snr}
        for dset in test dev; do
            dir=data/${dset}_${ns}${snr}
            echo "# --------> " ${dir}
            # create wav.scp
            python3 steps/data/augment_data_dir.py --bg-noise-dir data/${ns} --bg-snrs ${snr} data/${dset} data/${dset}_${ns}${snr}
            # spk2utt and utt2spk
            cp -v data/${dset}/spk2utt data/${dset}/utt2spk data/${dset}_${ns}${snr}/
            # create feats.scp
            steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" $dir
            steps/compute_cmvn_stats.sh $dir
        done
    done
done
# Alles wieder löschen
# find . -name "*_*noise*" -type d -exec rm -r "{}" \;
