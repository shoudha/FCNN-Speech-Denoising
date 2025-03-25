clc;
clearvars;
close all;

%This file tests the trained network.

% Select what type of noise
% 'mach' for machinery (non-speech) type noise
% 'babble' for speech type noise
noise_type = 'mach';

% Import Keras trained network
net = importKerasNetwork(strcat(noise_type, '_model.h5'),'OutputLayerType','regression');

% Load audio file
filename = strcat("Testing Audio/", noise_type, '_noise.wav');
[y, fsN] = audioread(filename);

% Initialization
overlap_size = 0.01*fsN; % 32 ms overlap
frame_size = 2*overlap_size; % 32 ms frame size
window = hanning(frame_size);

% Normilization
ch = y(:, 2);
ch = ch./max(ch);

% Resampling (Down-sampling)
fs2 = 16000;
ch = resample(ch, fs2, fsN); %resample 48 KHz to 16 KHz

% Discard extra samples from end
N_new = floor(length(ch)/(overlap_size/3))*(overlap_size/3);
ch = ch(1:N_new);

% Framing the audio
frames_ch = frame_sig(ch, frame_size/3, overlap_size/3, @hanning)';

% Test using trained network
predictors = reshape(frames_ch, [size(frames_ch, 1), 1, 1, size(frames_ch, 2)]);

%Latency estimation of the speech denoising
tic
denoisedFrames = predict(net, predictors);
toc

denoisedFrames = squeeze(denoisedFrames);

% Deframe
denoisedAudio = deframe_sig(denoisedFrames.', length(ch), frame_size/3, overlap_size/3, @hanning);

% Upsample
denoisedAudio = resample(denoisedAudio, fsN, fs2);

% There were some big spikes at the beginning and ending of the denoised
% audio. So these values were made zero wihout affecting the speech itself.
t_silence = .005; 
n_silence = t_silence*fsN;
denoisedAudio(1:n_silence) = 0;
denoisedAudio(end - n_silence:end) = 0;

% Volume matching with the test input
denoisedAudio = (max(y(:, 1))).*(denoisedAudio)./max(denoisedAudio);

% Plots
figure
subplot(221)

% Time axis of the testing audio
ty = 0:length(y(:, 1))-1;
ty = ty./fsN;

plot(ty, y(:, 2))
title("Example Speech Signal")
xlabel("Time (s)")
grid on
axis tight

subplot(223)

% Time axis of the denoised audio
tDy = 0:length(denoisedAudio)-1;
tDy = tDy./fsN;

plot(tDy, denoisedAudio)
title("Denoised Speech Signal")
xlabel("Time (s)")
grid on
axis tight

subplot(222)

% Frequency axis of the testing audio
freq_org = 48e3/2*linspace(-1/2,1/2,512);

plot(freq_org, db(fftshift(fft(y(:, 1), 512))))
title("Example Speech Signal Spectrum")
xlabel("Frequency (Hz)")
grid on
axis tight

subplot(224)

% Frequency axis of the denoised audio
plot(freq_org, db(fftshift(fft(denoisedAudio, 512))))
title("Denoised Speech Signal Spectrum")
xlabel("Frequency (Hz)")
grid on
axis tight















