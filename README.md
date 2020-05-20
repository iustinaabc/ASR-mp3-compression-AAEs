
# MP3 Compression as a Means to Improve Robustness against Adversarial Noise Targeting Attention-based End-to-End Speech Recognition

This work was performed as a Master's Thesis, which represents the last milestone for completing the Master’s Degree in Neuroengineering at the Technical University of Munich (TUM). It was conducted within the Department of Electrical and Computer Engineering, under the supervision of Univ.-Prof. Dr.-Ing. Bernhard U. Seeber (Chair of Audio Information Processing) and Dipl.-Ing.(Univ.) Ludwig K¨urzinger (Chair of Human-Machine Communication).

**Motivation & goal**

Adversarial Examples represent an imminent security threat to any Machine Learning system. In particular, Automatic Speech
Recognition (ASR) systems can be hacked into recognizing hidden voice commands delivered on purpose by a malicious external agent. In technical vocabulary, these are termed audio adversarial examples - they represent voice commands seemingly innocuous to a human listener, but which carry along a hidden message that can be decoded accordingly by the ASR system, and even passed on to a vocally-triggered execution system. The present thesis addresses this issue by proposing MP3-compression as a potential measure to reduce the susceptibility of Automatic Speech Recognition (ASR) systems to be mislead by Audio Adversarial Examples (AAEs).

**Methodology**

In essence, we used the Fast Gradient Sign Method (FGSM) to generate untargeted AAEs in the form of adversarial noise added to original speech samples. We used a feature inversion procedure to convert the adversarial examples from the feature into the audio domain. Different from prior work, we targeted an end-to-end, fully neural ASR system (namely ESPnet) featuring a hybrid decoder enhanced with both Connectionist Temporal Classification (CTC) and Attention mechanisms. Notably, this work did not focus on over-the-air, realtime attacks, but rather on direct attacks, in which adversarial samples were digitally
presented at the system’s input.

**Results**

* We found that MP3 compression applied to adversarial examples indeed reduces the recognition errors when compared to raw, uncompressed adversarial inputs. This result was validated by experiments with four ASR models trained on four types of audio data (uncompressed .wav format, as well as MP3 formats at three compression bitrates - 128, 64 and 24 kbps). 
* Additionally, when we decoded compressed adversarial examples originating from a different audio format than the training data, in a train-test mismatch scenario, we observed a further alleviation in the error rates. 
* In a parallel series of decoding experiments, we found that MP3 compression applied to speech inputs augmented with *non-adversarial noise* triggers an opposite behaviour of the ASR systems, in which more transcription errors are achieved than for uncompressed noise-augmented inputs. This finding consolidates the previous ones by suggesting that MP3 encoding is effective in diminishing only the adversarial noise. 
* Finally, a statistical test performed on the estimated Signal-to-Noise Ratio (SNR) of adversarial inputs confirmed that MP3-compressed adversarial samples had higher SNRs (hence less adversarial noise) than uncompressed adversarial inputs.


![Poster](/Poster_FINAL_18.05.2020.png)

Final [presentation](/FINAL_presentation_G-Drive_18.05.2020.pdf).

Complete experimental results can be found in `ASR_results.xlsx`
