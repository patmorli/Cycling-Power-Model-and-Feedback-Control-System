%{
ClosedloopAnalysis_v1, Patrick Mayerhofer, Mai, 2019
%}

clc; clear all ;

filename = 'Oct190_020_0_012CL_Patv2';
file = fullfile([filename  '.txt']);
fileID = fopen(file);
C = textscan(fileID,'%f %f %f %f %f %f %f', 'Delimiter',',');
fclose(fileID);
data = cell2mat(C);


%% get variables
time = data(:,3) - data(1, 3);
error = data(:,2);
gpsSpeed = data(:,1);
power = data(:, 4);
bpm = data(:, 6);
targetpower = data(:, 5);

%% delete all outliers, dont keep nans, delete outliers in time, bpm and gps as well
% [torque,torqueidx,torqueoutliers] = deleteoutliers(torquewd, 0.000001, 0);
% time(torqueidx) = [];
% targetpower(torqueidx) = [];
% gpsSpeedwd(torqueidx) = [];
% power(torqueidx) = [];
% error(torqueidx) = [];
% bpm(torqueidx) = [];

%% calculate rpm and power and bpm
for i = 1:length(time)-1
    rpm(i) = 30000/(time(i+1)-time(i));
end
time = time(2:end);
power = power(2:end);
targetpower = targetpower(2:end);
error = error(2:end);
gpsSpeed = gpsSpeed(2:end);
bpm = bpm(2:end);

fs = 3;
[b,a] = butter(4, .08*2/fs);
powerfilt = filtfilt(b,a,power);
rpmfilt = filtfilt(b,a,rpm);
bpmfilt = filtfilt(b,a,bpm);

 %% variance
 [rsquaredreg,msereg, rmsereg] = cost(targetpower,powerfilt)
 [rsquaredreg2 rmsereg2] = rsquare(targetpower,powerfilt)

%% plots 
figure(1);
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power & Cadence Data of Full Experiment', 'fontweight', 'bold')
yyaxis left; hold on; 
plot(time, rpm); plot(time, bpm); 
xlabel('Trialtime [s]', 'fontweight', 'bold'); ylabel('Cadence [bpm/rpm]', 'fontweight', 'bold');
yyaxis right; plot(time, power); plot(time, targetpower)
ylabel('Power [W]'); hold off;



figure(2); 
yyaxis left; hold on; plot(time); hold off;


figure(3); 
hold on;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Speed over Time', 'fontweight', 'bold')
plot(time, gpsSpeed); 
xlabel('Trialtime [s]', 'fontweight', 'bold'); ylabel('Speed [km/h]', 'fontweight', 'bold');

figure(4);
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Gains: Ki=0.01, Kp=0.01', 'fontweight', 'bold');
hold on; 
xlabel('Trialtime [s]', 'fontweight', 'bold'); ylabel('Power [W]', 'fontweight', 'bold');
plot(time, power, 'LineWidth',2); plot(time, targetpower, 'LineWidth',2);
plot(time, powerfilt, 'g', 'lineWidth', 3)
legend({'Actual Power','Target Power', 'Target Power Filtered'}, 'FontSize', 16);
%xlim([50000 600000])
hold off;

figure(4);
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Gains: Ki=0.02, Kp=0.012', 'fontweight', 'bold');
hold on; 
xlabel('Trialtime [s]', 'fontweight', 'bold'); ylabel('Power [W]', 'fontweight', 'bold');
plot(time, targetpower, 'LineWidth',2);
plot(time, powerfilt, 'lineWidth', 3)
legend({'Actual Power','Target Power', 'Target Power Filtered'}, 'FontSize', 16);
%xlim([50000 600000])
hold off;

figure(6);
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Gains: Ki=0.02, Kp=0.012', 'fontweight', 'bold');
hold on; 
xlabel('Trialtime [s]', 'fontweight', 'bold'); ylabel('Cadence [bpm/rpm]', 'fontweight', 'bold');
plot(time, rpm, 'LineWidth',2); plot(time, bpm, 'LineWidth',2);
legend({'RPM','BPM'}, 'FontSize', 16);
%xlim([50000 600000])
hold off;

time= time./1000;
figure(7);
subplot(2,1,1);
hold on;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Cadence', 'fontweight', 'bold')
plot(time(1470:2900)-time(1470), rpmfilt(1470:2900), 'LineWidth',2); plot(time(1470:2900)-time(1470), bpmfilt(1470:2900), 'LineWidth',2); 
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Cadence [rpm]', 'fontweight', 'bold');
hold off;

subplot(2,1,2); 
hold on;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power', 'fontweight', 'bold')
plot(time(1470:2900)-time(1470), targetpower(1470:2900), 'LineWidth',2);
plot(time(1470:2900)-time(1470), powerfilt(1470:2900), 'lineWidth', 3)
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Power [W]', 'fontweight', 'bold');
hold off;



% [linBpmrpmR linBpmrpmP] = corrcoef(bpm, rpm);
% [linBpmpowerR linBpmpowerP] = corrcoef(bpm, power);
% [linRpmpowerR linRpmpowerP] = corrcoef(rpm, power);
% 
% 
% 
% [cxyBPMRPM,fBPMRPM] = mscohere(bpm, rpm, [], [], [], 2.9);
% meancxyBPMRPM = mean(cxyBPMRPM);
% [cxyRPMPower,fRPMPower] = mscohere(rpm, power, [], [], [], 2.9);
% meancxyRPMPower = mean(cxyRPMPower);
% [cxyBPMPower,fBPMPower] = mscohere(bpm, rpm, [], [], [], 2.9);
% meancxyBPMPower = mean(cxyBPMPower);




%%
save(fullfile([filename  '.mat']));
%save('Openloop_Bike_v4_1');
