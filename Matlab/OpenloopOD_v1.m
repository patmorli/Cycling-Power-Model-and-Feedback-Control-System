%{
OpenloopOD_v1, Patrick Mayerhofer, Jul 27, 2018
Analyzes OD test chirp
change power calculation if changing magnets on crank
%}

clear all;
filename = 'Test21';

[~, ~, raw] = xlsread([filename  '.xlsx']); 
data = cell2mat(raw(3:end,:));

%% get rid of double measurements, maybe we dont need that anymore
datawd = data;

p = 1;
for i = 1:length(data)-1
    if data(i, 2) == data(i+1, 2)
        datawd(p+1, :) = [];
        p = p - 1;
    end
    p = p + 1;
end

%% get variables
time = datawd(:,2) - datawd(1, 2);
gpsSpeedwd = datawd(:,1);
torquewd = datawd(:, 3);
delfreqwd = datawd(:, 4);

%% delete all outliers, dont keep nans, delete outliers in time, bpm and gps as well
[torque,torqueidx,torqueoutliers] = deleteoutliers(torquewd, 0.000001, 0);
time(torqueidx) = [];
delfreqwd(torqueidx) = [];
gpsSpeedwd(torqueidx) = [];

%% calculate rpm and power and bpm
for i = 1:length(time)-1
    rpm(i) = 30000/(time(i+1)-time(i));
    bpm(i) = 60000/(delfreqwd(i) * 2);
    power(i) = torque(i+1) * (pi / ((time(i+1) - time(i)) / 1000));
end
time = time(2:end);
delfreq = delfreqwd(2:end);
torque = torque(2:end);
gpsSpeed = gpsSpeedwd(2:end);

% fs = 3;
% [b,a] = butter(4, .1*2/fs);
% powerfilt = filtfilt(b,a,power);

%% plots 
figure;
yyaxis left; hold on; plot(time(1:2850)/1000, rpm(1:2850), 'LineWidth', 2); plot(time(1:2850)/1000, bpm(1:2850), 'LineWidth', 2); 
ax = gca;
%ax.LabelFontSizeMultiplier = 3;
%ax.TitleFontSizeMultiplier = 3;
set(gca,'FontSize',20)
title('')
xlabel('Trialtime [s]') ;
ylabel('Cadence [BPM/RPM]'); 
yyaxis right; plot(time(1:2850)/1000, power(1:2850), 'LineWidth', 2);
ylabel('Power [W]') 
legend({'RPM','BPM','Power'}, 'FontSize', 16);

figure; yyaxis left; hold on; plot(time(1:2850))
figure; plot(time(1:2850), gpsSpeed(1:2850))

[linBpmrpmR linBpmrpmP] = corrcoef(bpm, rpm);
[linBpmpowerR linBpmpowerP] = corrcoef(bpm, power);
[linRpmpowerR linRpmpowerP] = corrcoef(rpm, power);

[cxyBPMRPM,fBPMRPM] = mscohere(bpm, rpm, [], [], [], 2.9);
meancxyBPMRPM = mean(cxyBPMRPM);
[cxyRPMPower,fRPMPower] = mscohere(rpm, power, [], [], [], 2.9);
meancxyRPMPower = mean(cxyRPMPower);
[cxyBPMPower,fBPMPower] = mscohere(bpm, rpm, [], [], [], 2.9);
meancxyBPMPower = mean(cxyBPMPower);




%%
save(filename);
%save('Openloop_Bike_v4_1');
