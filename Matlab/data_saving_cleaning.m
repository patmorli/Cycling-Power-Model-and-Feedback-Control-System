%{
OpenloopOD_v1, Patrick Mayerhofer, Oct, 2018
Analyzes OD test chirp
change power calculation if changing magnets on crank
%}

clc; close; clear all ;
%% changeable variables
subject_id = 10;
trial_id = 1;
save_flag = 0;

rw = 0.3; %works best with actual gpsspeed
%rw=0.325; %radius wheel original
lca = 0.17; %length crank arm
fbracket = 39;

%% import
filename = ['Subject', num2str(subject_id), '_', num2str(trial_id)];
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/';
dir_load_file = [dir_root, 'OpenLoop/RawData/', filename, '.txt'];
dir_save_file = [dir_root, 'OpenLoop/MatFiles/', filename, '.mat'];
data = readtable(dir_load_file);
data = table2array(data);
% for weight and gear
dir_summary_file = [dir_root, 'Summary.xlsx'];
summary_file = readtable(dir_summary_file);
local_summary_file = summary_file; % to change things before saving again

%% get constants from summary file
m = summary_file.Weight(subject_id);

%% store rbracket to summary file
[local_summary_file, rbracket] = gear_conversion(local_summary_file, summary_file.defaultGear(subject_id), trial_id, subject_id);

GR = fbracket/rbracket;


m = m + 12.6; % bike plus equipment 



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
gpsSpeed = gpsSpeed/3.6;
time = time/1000;


fs = 3;
[b,a] = butter(4, .08*2/fs);
P = filtfilt(b,a, power);


%% plots, all without time to find the right index if needs deleted
figure(1);
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power & Cadence Data of Full Experiment', 'fontweight', 'bold')
yyaxis left; hold on; 
plot(rpm); plot(bpm); 
xlabel('Index', 'fontweight', 'bold'); ylabel('Cadence [bpm/rpm]', 'fontweight', 'bold');
yyaxis right; plot(P);
ylabel('Power [W]'); hold off;

%% calculate filtered and derivatives
% filter speed and rpm
fs = 3;
[b,a] = butter(4, .1*2/fs);
gpsSpeed_filtered = filtfilt(b,a,gpsSpeed);
rpm_filtered = filtfilt(b,a,rpm);

%% speed from cadence
vcalc = (rw * 2*pi*GR.*rpm_filtered)/60;

% derivative of gpsspeed, vcalc, and rpm
dv = nanmean([diff([gpsSpeed_filtered' NaN]); diff([NaN gpsSpeed_filtered'])]); % central difference
dt = nanmean([diff([time' NaN]); diff([NaN time'])]);
vdot = dv./dt;

dvcalc = nanmean([diff([vcalc NaN]); diff([NaN vcalc])]); % central difference
vcalcdot = dvcalc./dt;

rpm_filtered = rpm_filtered';
drpm = nanmean([diff([rpm_filtered' NaN]); diff([NaN rpm_filtered'])]); % central difference
rpmdot_filtered = drpm./dt;

% plot
figure(2); 
hold on;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Speed over Time', 'fontweight', 'bold')
plot(gpsSpeed_filtered); 
plot(vcalc);
xlabel('Index', 'fontweight', 'bold'); ylabel('Speed [km/h]', 'fontweight', 'bold');

%% here we will fill in the parts that need to be deleted, and then it deletes 
prompt = 'Which parts do you want to delete you awesome scientist? Fill in in this form: [1:100, length(vcalc)-100:length(vcalc)]';
delete = input(prompt);
%create an array of ones
include_index = ones(length(time),1);
% make all the ones that we chose 0
for i = 1:length(delete)
    include_index(delete(i)) = 0; 
end

%just include the ones that stayed 1
clean_gpsSpeed = gpsSpeed(include_index==1);
clean_bpm = bpm(include_index==1)';
clean_rpm = rpm(include_index==1)';
clean_power = power(include_index==1)';
clean_time = time(include_index==1);
clean_gpsSpeed_filtered = gpsSpeed_filtered(include_index==1);
clean_vdot_filtered = vdot(include_index==1)';
clean_rpm_filtered = rpm_filtered(include_index==1);
clean_rpmdot_filtered = rpmdot_filtered(include_index==1)';
clean_vcalc = vcalc(include_index==1)';
clean_vcalcdot = vcalcdot(include_index==1)';
close all

% filter power 
clean_power_filtered = filtfilt(b,a,clean_power);


% create a table
csvDocument = table(clean_time, clean_bpm, clean_rpm, clean_rpm_filtered, clean_rpmdot_filtered, clean_gpsSpeed,clean_gpsSpeed_filtered, clean_vdot_filtered, clean_vcalc, clean_vcalcdot, clean_power, clean_power_filtered, 'VariableNames',{'time','bpm', 'rpm','rpm_filtered', 'rpmdot_filtered', 'gpsSpeed', 'gpsSpeed_filtered', 'gpsSpeeddot_filtered', 'vcalc_filtered', 'vcalcdot_filtered' ,'power', 'power_filtered'});
% plot again
figure(3);
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Power & Cadence Data of Full Experiment', 'fontweight', 'bold')
yyaxis left; hold on; 
plot(csvDocument.rpm); plot(csvDocument.bpm); 
xlabel('Index', 'fontweight', 'bold'); ylabel('Cadence [bpm/rpm]', 'fontweight', 'bold');
yyaxis right; plot(csvDocument.power_filtered);
ylabel('Power [W]'); hold off;

figure(4); 
hold on;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Speed over Time', 'fontweight', 'bold')
plot(csvDocument.gpsSpeed_filtered); 
plot(csvDocument.vcalc_filtered)
xlabel('Index', 'fontweight', 'bold'); ylabel('Speed [km/h]', 'fontweight', 'bold');

disp('are you sure you want to save this? Click enter if yes. ');
pause();

%% 
% split in train and test set
train_length = round(height(csvDocument)*0.7);
csvDocument_train = csvDocument(1:train_length, :);
csvDocument_test = csvDocument(train_length+1:height(csvDocument), :);


dir_save_file_train = [dir_root, 'OpenLoop/CleanedCSV/', filename, '_train.csv'];
dir_save_file_test = [dir_root, 'OpenLoop/CleanedCSV/', filename, '_test.csv'];




%%
if save_flag
    writetable(csvDocument_train, dir_save_file_train);
    writetable(csvDocument_test, dir_save_file_test);
    writetable(local_summary_file, dir_summary_file);
    save(dir_save_file);
end

close all;
