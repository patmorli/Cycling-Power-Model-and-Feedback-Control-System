% wind_analysis_together
% December, 2021, Patrick Mayerhofer
clear all; close all;

s_id = [3:13];
dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/OpenLoop/';

wind_all = [];
for subject_id = s_id
    %% load and analyze wind data
    dir_wind_file = [dir_root, 'WindData/Summary_sub', num2str(subject_id), '.xlsx'];
    wind = readtable(dir_wind_file);
    wind_all = [wind_all; wind];
    
end
wind = [];
wind.mean = mean(wind_all.mean);
wind.std = mean(wind_all.std);

wind = struct2table(wind);

dir_save = [dir_root, 'WindData/Summary.xlsx'];
writetable(wind, dir_save);
