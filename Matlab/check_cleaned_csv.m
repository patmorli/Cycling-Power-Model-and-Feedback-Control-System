%{
check_cleaned_csv, Patrick Mayerhofer, Jan, 2022
to plot different parts of the cleaned csv files
%}
clear all; close all;
subject_id = 8;
trial_id = 1;
plot_gps_vs_vcalc = 1;
plot_vcalcdot = 1;

dir_root = '/Volumes/GoogleDrive/My Drive/Cycling Project/2021/SubjectData/';
filename = ['Subject', num2str(subject_id), '_', num2str(trial_id)];
dir_load_file = [dir_root, 'OpenLoop/CleanedCSV/', filename, '.csv'];
data = readtable(dir_load_file);

if plot_gps_vs_vcalc
    figure(); 
    hold on;
    ax = gca;
    ax.LabelFontSizeMultiplier = 2;
    ax.TitleFontSizeMultiplier = 2;
    title('Speed', 'fontweight', 'bold')
    plot(data.time*1000, data.gpsSpeed_filtered); 
    plot(data.time*1000, data.vcalc_filtered);
    xlabel('Trialtime [s]', 'fontweight', 'bold'); ylabel('Speed [km/h]', 'fontweight', 'bold');
end

if plot_vcalcdot
    figure(); 
    hold on;
    ax = gca;
    ax.LabelFontSizeMultiplier = 2;
    ax.TitleFontSizeMultiplier = 2;
    title('Acceleration', 'fontweight', 'bold')
    plot(data.time*1000, data.vcalcdot_filtered); 
    xlabel('Trialtime [s]', 'fontweight', 'bold'); ylabel('Acceleration [m/s^2]', 'fontweight', 'bold');
end