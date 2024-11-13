
Fs = 2.9; %mean of sample frequency
Y=fft(power); %fft
P2 = abs(Y/length(Y)); %two-sided spectrum
P1 = P2(1:length(Y)/2+1); %single-sided spectrum 
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(length(Y)/2))/length(Y); %frequencies that are in the signal
figure;
plot(f,P1)
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')




Y2=fft(bpm);
P22 = abs(Y2/length(Y2));
P12 = P22(1:length(Y2)/2+1);
P12(2:end-1) = 2*P12(2:end-1);
f2 = Fs*(0:(length(Y2)/2))/length(Y2);
figure;
plot(f2,P12)
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P12(f)|')
    

