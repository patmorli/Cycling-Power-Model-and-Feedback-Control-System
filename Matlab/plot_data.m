% plot_data, Patrick Mayerhofer, October 2021
% plots the data of the specified subject and trial
clear all; close all;

%% load data
subject_id = 1;
trial_id = 3;
fold = 3; % because we do 5 fold testing
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/ResultMatFiles/Pilot/';
filename = [dir_root, 'Subject', num2str(subject_id), '_', num2str(trial_id), '.mat'];
load(filename);

fig_test = figure();
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Predicted Power vs Actual Power normalized-test')
xlabel('Trialtime [s]') 
ylabel('Power [W]') 
hold on;
plot(power_measured_norm_test{fold}, 'b', 'LineWidth', 3);
plot(power_predicted_norm_test{fold}, 'r', 'LineWidth', 3);
legend({'Measured','Predicted'}, 'FontSize', 16);

fig_train = figure();
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('Predicted Power vs Actual Power normalized-train')
xlabel('Trialtime [s]') 
ylabel('Power [W]') 
hold on;
plot(power_measured_norm_train{fold}, 'b', 'LineWidth', 3);
plot(power_predicted_norm_train{fold}, 'r', 'LineWidth', 3);
legend({'Measured','Predicted'}, 'FontSize', 16);

