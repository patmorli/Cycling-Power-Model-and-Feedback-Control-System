% make_full_trial, Patrick Mayerhofer, October 2021
% this is a script, because I made a mistake before. 
% it takes the train and test file of each cycling trial
% and puts it into one single file.

%% variables
subject_id = 1;
t_id = 3;
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/CleanedCSV/';

for s_id = subject_id
    for trial_id = t_id
        %% load files
        filename_test = ['Subject', num2str(s_id), '_', num2str(trial_id), '_test.csv'];
        filename_train = ['Subject', num2str(s_id), '_', num2str(trial_id), '_train.csv'];
        filename = ['Subject', num2str(s_id), '_', num2str(trial_id), '.csv'];

        file_test = readtable([dir_root, filename_test]);
        file_train = readtable([dir_root, filename_train]);

        %% put together
        file = [file_train;file_test];
%         
%         %% create file that specifies 5 folds for cross validation 
%         l = size(file,1);
%         l5 = round(l/5);
%         increments = [1, l5, l5*2, l5*3, l5*4, l];

        %% save
        writetable(file, [dir_root, filename]);
    end
end
