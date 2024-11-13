%{
analyze_pilot, Patrick Mayerhofer, Nov, 2021
takes all data from the 4 pilot subjects and puts into an excel sheet
%}
clear all; close all;

%% 
ids = {'1_1'; '1_3'; '8_1'; '8_2'; '11_1'; '11_2'; '11_3'; '12_1'; '12_2'};

%% loop over each sub_trial
for s = 1:length(ids)
    % load here because it will be overwritten otherwise
    dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/ResultMatFiles/Combined/Pilot/';
    load([dir_root, 'Subject', ids{s}, '.mat']);
    r2_train_mean(s) = mean(r2_norm_train);
    r2_train_std(s) = std(r2_norm_train);
    r2_test_mean(s) = mean(r2_norm_test);
    r2_test_std(s) = std(r2_norm_train);
    
    rmse_train_mean(s) = mean(rmse_norm_train);
    rmse_train_std(s) = std(rmse_norm_train);
    rmse_test_mean(s) = mean(rmse_norm_test);
    rmse_test_std(s) = std(rmse_norm_train);
end

data = table(ids, r2_test_mean', r2_test_std', r2_train_mean', r2_train_std', rmse_test_mean', rmse_test_std', rmse_train_mean', rmse_train_std',...
    'VariableNames', {'ids', 'r2_test_mean', 'r2_test_std', 'r2_train_mean', 'r2_train_std', 'rmse_test_mean', 'rmse_test_std', 'rmse_train_mean', 'rmse_train_std'});

% save
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/ResultMatFiles/Pilot/';
writetable(data, [dir_root, 'Pilot_Summary.csv']);