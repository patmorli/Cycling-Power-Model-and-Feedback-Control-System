%{
OpenloopOD_v1, Patrick Mayerhofer, Oct, 2018
Analyzes OD test chirp
change power calculation if changing magnets on crank
%}

clc; close; clear all ;
filename = 'Subject21_1';
save_flag = 0;

fbracket = 39;
rbracket = 17;
GR = fbracket/rbracket;

rw=0.325; %radius wheel
lca = 0.17; %length crank arm
m = 66;
m = m + 8; % find out actual weight of bike. 

%dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SpeedEval/';
dir_root = '/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Cycling Project/2021/SubjectData/OpenLoop/';
dir_load_file = [dir_root, 'RawData/', filename, '.txt'];
%dir_load_file = [dir_root, filename, '.txt'];
dir_save_file = [dir_root, 'MatFiles/', filename, '.mat'];
data = readtable(dir_load_file);
data = table2array(data);
% file = [filename,  '.txt'];
% fileID = fopen(file);
% C = textscan(file,'%f %f %f %f', 'Delimiter',',');
% fclose(fileID);
% data = cell2mat(C);

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

fs = 3;
[b,a] = butter(4, .08*2/fs);
P = filtfilt(b,a, power);

rpm_filtered = filtfilt(b,a, rpm);

% %% speed from cadence
%  vcalc = (rw * 2*pi*GR.*rpm_filtered)/60;

%% plots 
figure(1);
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power & Cadence Data of Full Experiment', 'fontweight', 'bold')
yyaxis left; hold on; 
plot(time, rpm); plot(time, bpm); 
xlabel('Trialtime [s]', 'fontweight', 'bold'); ylabel('Cadence [bpm/rpm]', 'fontweight', 'bold');
yyaxis right; plot(time, P);
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
subplot(2,1,1);
hold on;
% ax = gca;
% ax.LabelFontSizeMultiplier = 2;
% ax.TitleFontSizeMultiplier = 2;
title('Cadence', 'fontweight', 'bold')
plot(time, rpm_filtered); plot(time, bpm); 
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Cadence [bpm/rpm]', 'fontweight', 'bold');

subplot(2,1,2); 
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power', 'fontweight', 'bold')
plot(time, P);
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Power [W]', 'fontweight', 'bold');

% calculates the correlation between the two. 
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
if save_flag
    save(dir_save_file);
end
