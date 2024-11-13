function [f, P1, noiseamp] = noise_v1(signal,time)
    % this function calculates the noise of a signal
    % created June 2019, Patrick Mayerhofer
    % needs signal, time (s) of datapoints
    % evaluates mean sampling frequency and then does fft
   
    %% frequency analysis
    % signal length
    L = length(signal);
    
    % mean sampling frequency
    for i = 1:length(time)-1
        diftime(i) = time(i+1)-time(i); %time dif of each sample
    end
    meandiftime = mean(diftime); 
    Fs = 1/meandiftime; %mean sampling frequency of signal
    
    %Fs = 10; %Sampling Frequency
    T = 1/Fs;
    %L = 1610; %Signal length
    signal = signal - mean(signal);
    Y = fft(signal);
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:(L/2))/L;
%     plot(f,P1);
%     title('Single-Sided Amplitude Spectrum of X(t)')
%     xlabel('f (Hz)')
%     ylabel('|P1(f)|')
    
    %% amplitude analysis
    for i = 1:length(signal)-1
        difsignal(i) = signal(i+1)-signal(i); %time dif of each sample
    end
    noiseamp = mean(abs(difsignal)); %noiseamplitude
end

