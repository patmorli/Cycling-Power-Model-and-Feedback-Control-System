%% delete weird parts, and beginning, and save rest as .csv
clear; close;
filename = 'Subject10_1';




dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/';
dir_load_file = [dir_root, 'MatFiles/', filename, '.mat'];
load(dir_load_file);
time = time/1000;
gpsSpeed = gpsSpeed/3.6;

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
prompt = 'Which parts do you want to delete you awesome scientist? Fill in in this form: [1:10, 510:600] ';
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
yyaxis right; plot(csvDocument.power);
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


dir_save_file_train = [dir_root, 'CleanedCSV/', filename, '_train.csv'];
dir_save_file_test = [dir_root, 'CleanedCSV/', filename, '_test.csv'];


writetable(csvDocument_train, dir_save_file_train);
writetable(csvDocument_test, dir_save_file_test);


