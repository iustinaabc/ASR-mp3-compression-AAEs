
# MP3 Compression To Diminish Adversarial Noise in End-to-End Speech Recognition

## TL;DR
This work explored __MP3 compression__ as a countermeasure to _Audio Adversarial Examples (AAEs)_ - the hostile inputs that trick Automatic Speech Recognition (ASR) systems into recognizing hidden commands, while human users remain oblivious to their presence. The malicious character of these inputs is usually given by a specially crafted noise (_Adversarial Noise_) that is added to regular audio inputs.
<p align="center">
  <img src="/ASR_real-life_attack_with_TV.jpg" alt="ASR_attack" width="500px" height="250px"><br>
  (Adapted from Schönherr et al. 2019)
</p>

To prevent this kind of attack, we implemented the following pipeline: we generated AAEs with the _Fast Gradient Sign Method (FGSM)_ for an end-to-end, hybrid CTC-attention ASR system. We then performed decoding experiments with both uncompressed and MP3-compressed AAEs, and validated the presence of Adversarial Noise with two objective indicators: <br>
(1) Character Error Rates (CER) that measure the speech decoding performance of four ASR models trained on uncompressed, as well as MP3-compressed data sets;<br>
(2) Signal-to-Noise Ratio (SNR) estimated for both uncompressed and MP3-compressed AAEs that are reconstructed in the time domain by feature inversion. 

## Spoilers
* You might first want to indulge your ears in these delightful [adversarial audio samples](/audio_samples). We give away free (German!) beers to those who can recognize the hidden phrases. 
* For the detail-cravers, take a look at the complete experimental results in `ASR_results.xlsx`
* If curious about the technicalities, feel free to consult the [presentation](/FINAL_presentation_G-Drive_18.05.2020.pdf) that was delivered for the Masters' Thesis completion.


## Detailed description

This work was performed as a Master's Thesis and represents the last milestone for completing the Master's Degree in Neuroengineering at the Technical University of Munich (TUM). It was conducted from October 2019 to May 2020 within the Department of Electrical and Computer Engineering, under the supervision of Univ.-Prof. Dr.-Ing. Bernhard U. Seeber (Chair of Audio Information Processing) and Dipl.-Ing.(Univ.) Ludwig Kürzinger (Chair of Human-Machine Communication). 


**Motivation & goal**

**Adversarial Examples** represent an imminent security threat to any Machine Learning system. In particular, Automatic Speech
Recognition (ASR) systems can be hacked into recognizing hidden voice commands delivered on purpose by a malicious external agent. In technical vocabulary, these are termed **Audio Adversarial Examples (AAEs)** - they represent voice commands seemingly innocuous to a human listener, but which carry along a hidden message that can be decoded accordingly by the ASR system, and even passed on to a vocally-triggered execution system. The present work addresses this issue by proposing MP3-compression as a potential measure to reduce the susceptibility of Automatic Speech Recognition (ASR) systems to be mislead by Audio Adversarial Examples (AAEs).

**Methodology**

We used the Fast Gradient Sign Method (FGSM) to generate untargeted AAEs in the form of __Adversarial Noise__ added to original speech samples. We used a feature inversion procedure to convert the adversarial examples from the feature into the audio domain. Different from prior work, we targeted an end-to-end, fully neural ASR system (namely ESPnet) featuring a hybrid decoder enhanced with both Connectionist Temporal Classification (CTC) and Attention mechanisms. Notably, this work did not focus on over-the-air, realtime attacks, but rather on direct attacks, in which adversarial samples were digitally
presented at the system's input.

**Results**

* We found that MP3 compression applied to adversarial examples indeed reduces the recognition errors when compared to raw, uncompressed adversarial inputs. This result was validated by experiments with four ASR models trained on four types of audio data (uncompressed .wav format, as well as MP3 formats at three compression bitrates - 128, 64 and 24 kbps). 
* Additionally, when we decoded compressed adversarial examples originating from a different audio format than the training data, in a train-test mismatch scenario, we observed a further alleviation in the error rates. 
* In a parallel series of decoding experiments, we found that MP3 compression applied to speech inputs augmented with *non-adversarial noise* triggers an opposite behaviour of the ASR systems, in which more transcription errors are achieved than for uncompressed noise-augmented inputs. This finding consolidates the previous ones by suggesting that MP3 encoding is effective in diminishing only the adversarial noise. 
* Finally, a statistical test performed on the estimated Signal-to-Noise Ratio (SNR) of adversarial inputs confirmed that MP3-compressed adversarial samples had higher SNRs (hence less adversarial noise) than uncompressed adversarial inputs.

**Preliminary Poster**
![Poster](/Poster_FINAL_18.05.2020.png)

# Reference

This work was published at _SPECOM 2020_ at https://doi.org/10.1007/978-3-030-60276-5_3 and can also be found in the [__arXiv preprint__](https://arxiv.org/abs/2007.12892). To cite this work:

```
@InProceedings{10.1007/978-3-030-60276-5_3,
author="Andronic, Iustina
and K{\"u}rzinger, Ludwig
and Chavez Rosas, Edgar Ricardo
and Rigoll, Gerhard
and Seeber, Bernhard U.",
editor="Karpov, Alexey
and Potapova, Rodmonga",
title="MP3 Compression to Diminish Adversarial Noise in End-to-End Speech Recognition",
booktitle="Speech and Computer",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="22--34",
abstract="Audio Adversarial Examples (AAE) represent purposefully designed inputs meant to trick Automatic Speech Recognition (ASR) systems into misclassification. The present work proposes MP3 compression as a means to decrease the impact of Adversarial Noise (AN) in audio samples transcribed by ASR systems. To this end, we generated AAEs with a new variant of the Fast Gradient Sign Method for an end-to-end, hybrid CTC-attention ASR system. The MP3's effectiveness against AN is then validated by two objective indicators: (1) Character Error Rates (CER) that measure the speech decoding performance of four ASR models trained on different audio formats (both uncompressed and MP3-compressed) and (2) Signal-to-Noise Ratio (SNR) estimated for uncompressed and MP3-compressed AAEs that are reconstructed in the time domain by feature inversion. We found that MP3 compression applied to AAEs indeed reduces the CER when compared to uncompressed AAEs. Moreover, feature-inverted (reconstructed) AAEs had significantly higher SNRs after MP3 compression, indicating that AN was reduced. In contrast to AN, MP3 compression applied to utterances augmented with regular noise resulted in more transcription errors, giving further evidence that MP3 encoding is effective in diminishing AN exclusively.",
isbn="978-3-030-60276-5"
}
```

The method to generate audio adversarial examples that was used in this work is described at https://doi.org/10.1007/978-3-030-60276-5_26.
