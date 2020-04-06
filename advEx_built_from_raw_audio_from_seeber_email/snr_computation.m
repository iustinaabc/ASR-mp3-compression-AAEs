% SNR computation between original wavs and adversarial samples
utts=["2old2play-20110606-hcn-a0297", "akhansson-20120423-upv-a0138", ...
    "zlp-20100110-kgx-a0330", "zlp-20100110-wwn-b0204"];
for i=1:4
[orig_wav fs] = audioread(strcat("0_original_raw_audio/", utts(i),".wav"));
[reconstr_orig ~] = audioread(strcat("2_reconstructed_raw_audio_from_0(from_nonAdvEx_feats)/", utts(i), "_recon.wav"));
[advEx_wav ~] = audioread(strcat("3_advEx_audio_reconstructed_from_advEx_feats/", utts(i), "_advEx.wav"));
[compr_advEx_24kbps ~] = audioread(strcat("6_advEx_audio_from_3_mp3-compressed_24kbps/", utts(i), "_advEx_24kbps.wav"));

figure
subplot(2,2,1)
plot(orig_wav); ylim([-1 1]);
subplot(2,2,2)
plot(reconstr_orig); ylim([-1 1]);
subplot(2,2,3)
plot(advEx_wav); ylim([-1 1]);
subplot(2,2,4)
plot(compr_advEx_24kbps); ylim([-1 1]);
suptitle(utts(i))

% Directly substract the orig signal from the adversarial audio
%  advers_noise = advEx_wav(1:end,1) - orig_wav(1:length(advEx_wav),1);
%  advers_noise_abs = abs(advEx_wav(1:end,1)) - abs(orig_wav(1:length(advEx_wav),1));

 advers_noise = advEx_wav(1:end,1) - reconstr_orig(1:length(advEx_wav),1);
 advers_noise_abs = abs(advEx_wav(1:end,1)) - abs(reconstr_orig(1:length(advEx_wav),1));
 
% figure
% plot(advers_noise)
% IA: Actually, it's wrong to do this substraction !!

%  sound(orig_wav, fs);
%  sound(advEx_wav, fs);
%  sound(advers_noise, fs); % this is almost the same as advEx_wav !!
%  sound(advers_noise_abs, fs); % we cannot hear exactly the adversarial noise !
% play(audioplayer(advers_noise, fs)); % doesn't work !!

% pow_sig = sum(orig_wav.^2);
pow_sig(i,1) = sum(orig_wav.^2);
pow_sig_recon(i,1) = sum(reconstr_orig.^2);
pow_advEx(i,1) = sum(advEx_wav.^2);
pow_advEx_NOISE(i,1) = pow_advEx(i,1) - pow_sig(i,1);
snr10(i,1) = 10 * log10 (pow_sig(i,1)/pow_advEx_NOISE(i,1));

snr20(i,1) = 20 * log10 (pow_sig(i,1)/pow_advEx_NOISE(i,1));
%snr20 = 20 * log10 (pow_sig/pow_noise)
end


