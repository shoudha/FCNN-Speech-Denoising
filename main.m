clc;
clearvars;
close all;

%This file creates training variables as .mat file from the recorded audio files and performs
%VAD on them and saves them in their respective directories. 

% Select what type of noise
% 'mach' for machinery (non-speech) type noise
% 'babble' for speech type noise
noise_type = 'mach';

% Folder and filenames of training audio files
folder = 'Training Audio/';
filename = strcat(folder, 'speech_with_', noise_type, '_noise.wav');

[y, fsN] = audioread(filename);

% Initialization
overlap_size = 0.01*fsN; % 10 ms overlap
frame_size = 2*overlap_size; % 20 ms frame size
window = hanning(frame_size);

% Normilization
ch1 = y(:, 1);
ch2 = y(:, 2);

ch1 = ch1./max(ch1);
ch2 = ch2./max(ch2);

% Resampling (Down-sampling)
fs2 = 16000;
ch1 = resample(ch1, fs2, fsN); %resample 48 KHz to 16 KHz
ch2 = resample(ch2, fs2, fsN); %resample 48 KHz to 16 KHz

% Discard extra samples from end
N_new = floor(length(ch1)/(overlap_size/3))*(overlap_size/3);
ch1 = ch1(1:N_new);
ch2 = ch2(1:N_new);

% Framing the audio
frames_ch1 = frame_sig(ch1, frame_size/3, overlap_size/3, @hanning)';
frames_ch2 = frame_sig(ch2, frame_size/3, overlap_size/3, @hanning)';

% Voice Activity Detection 
% Initialize VAD parameters
VAD_cst_param = vadInitCstParams;
VAD_cst_param.Fs = fs2;
VAD_cst_param.L_NEXT = overlap_size/3;
VAD_cst_param.L_FRAME = frame_size/3;
VAD_cst_param.hamwindow = hanning(frame_size); % size = 3*L_FRAME
VAD_cst_param.L_WINDOW = length(VAD_cst_param.hamwindow); % size = 3*L_FRAME

decision = zeros(1, size(frames_ch1, 2));
for i = 1:size(frames_ch1, 2)
    clear vadG729
    decision(i) = vadG729(frames_ch1(:, i), VAD_cst_param); % should feed frame-base
end

% Extract the Voice frames
indx_NS = find(decision == 1);
targets = frames_ch1(:, indx_NS);
predictors = frames_ch2(:, indx_NS);

% Save the framed audio signals in their rexpective directories
save('targets_for_training.mat', 'targets')
save('predictors_for_training.mat', 'predictors')

%Plots
%Expanding the decision boundaries from frames to 
%entire audio signal to visualize the VAD performance
decision_ext = [decision(1) decision];
decision_ext = repmat(decision_ext, overlap_size/3, 1);
decision_ext = decision_ext(:);

time = (1:length(ch1))./16e3;

subplot(211)
plot(time, ch1)
title('Channel 1')
xlabel('time(s)')
ylabel('Amplitude')

hold on
plot(time, decision_ext)
axis tight

subplot(212)
plot(time, ch2)
title('Channel 2')
xlabel('time(s)')
ylabel('Amplitude')

hold on
plot(time, decision_ext)
axis tight







