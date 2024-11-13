%{
Baselinecheck_v2
Started: Apr. 9, 2018
modified: April, 2021
Calculates mean bpm during a 30 seconds window happening before the last 30 seconds, 
mean torque (not needed) deletes outliers, plots data
%}
clear all;

dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/BaseCadence/'
filename = 'Subject3'
data = readtable([dir_root, filename,  '.csv']); 


%% get time and torque

time1 = data.Var2;
torque1 = data.Var3;

%% get rid of double measurements, maybe we dont need that anymore
timewd = time1; %wd..without double
torquewd = torque1;
p = 1;
for i = 1:length(time1)-1
    if time1(i) == time1(i+1)
        timewd(p+1) = [];
        torquewd(p+1) = [];
        p = p - 1;
    end
    p = p + 1;
end


%% get rid of 0s and set time back
torquewd(torquewd == 0) = NaN;
time = timewd - timewd(1);


%% calculate rpm and power
for i = 1:length(time)-1
    rpmwd(i) = (60000/(time(i+1)-time(i)))/2;
    powerwd(i) = torquewd(i+1) * (2*pi / ((time(i+1) - time(i)) / 1000));
end
time = time(2:end);
%% delete all outliers, keep nans, calculate mean and std of 30 seconds, 
% 30 seconds before stop

%outliers
[rpm,rpmidx,rpmoutliers] = deleteoutliers(rpmwd, 0.05, 1);
[power,poweridx,poweroutliers] = deleteoutliers(powerwd, 0.05, 1);

% area of interest
time30 = time<=time(end)-30000 & time>=time(end)-60000;

% rpm
rpmMean = nanmean(rpm(time30));
rpmStd = nanstd(rpm(time30));

% power
powerMean = nanmean(power(time30));
powerStd = nanstd(power(time30));

%%



plot(time, rpmwd, 'r');
hold on;
%axis([0 inf 60 100])
plot(time, rpm, 'b');

t = time(time30);
line([t(1), t(end)], [rpmMean, rpmMean], 'Color', 'blue', 'LineWidth', 3, 'LineStyle', '--');

figure(2);
subplot(2,1,1);
hold on;
% ax = gca;
% ax.LabelFontSizeMultiplier = 2;
% ax.TitleFontSizeMultiplier = 2;
title('Cadence', 'fontweight', 'bold')
plot(time/1000, rpm); 
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Cadence [bpm/rpm]', 'fontweight', 'bold');

subplot(2,1,2); 
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power', 'fontweight', 'bold')
plot(time/1000, power);
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Power [W]', 'fontweight', 'bold');


fs = 3;
[b,a] = butter(4, .08*2/fs);
P = filtfilt(b,a, power(208:429));

rpm_filtered = filtfilt(b,a, rpm(208:429));

figure(3);
subplot(2,1,1);
hold on;
% ax = gca;
% ax.LabelFontSizeMultiplier = 2;
% ax.TitleFontSizeMultiplier = 2;
title('Cadence', 'fontweight', 'bold')
plot(time(208:429)/1000, rpm_filtered); 
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Cadence [bpm/rpm]', 'fontweight', 'bold');

subplot(2,1,2); 
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power', 'fontweight', 'bold')
plot(time(208:429)/1000, P);
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Power [W]', 'fontweight', 'bold');


%save(filename);



