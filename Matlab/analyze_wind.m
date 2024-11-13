% wind_analysis
% December, 2021, Patrick Mayerhofer
clear all; close all;

s_id = [13];
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/';


for subject_id = s_id
    %% load and analyze wind data
    dir_wind_file = [dir_root, 'WindData/Subject', num2str(subject_id), '.xlsx'];
    wind_file = readtable(dir_wind_file);
    
    % find starts and ends of the trials
    trial_starts = find(wind_file.Var1 == 0);
    trial_ends = find(isnan(wind_file.Var1) == 1); % finds all nans. 
    % there are more nans then only at the end of the trial. need to
    % specify
    
    trial_ends = trial_ends(trial_ends > trial_starts(1));
    
    trial{1} = wind_file(trial_starts(1):trial_ends(1)-1,:);
    trial{2} = wind_file(trial_starts(2):trial_ends(2)-1,:);
    trial{3} = wind_file(trial_starts(3):end,:);

    
    for i = 1:3
        wind(i).mean = nanmean(trial{i}.current);
        wind(i).std = nanstd(trial{i}.current);
        wind(i).max = max(trial{i}.current);
        wind(i).min = min(trial{i}.current);
    end
    
    wind = struct2table(wind);
    
    dir_save = [dir_root, 'WindData/Summary_sub', num2str(subject_id), '.xlsx'];
    writetable(wind, dir_save);
end

