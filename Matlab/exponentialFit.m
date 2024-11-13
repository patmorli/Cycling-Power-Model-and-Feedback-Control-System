%% exponential fit, Patrick Mayerhofer, October, 2019
%for real trials
x=mean(timetrialsSpline,1)';
y=mean(powerfilttrialsNormSpline,1)';
x1_2=x(101:end)-x(101);
y1_2=y(101:end)-1.01;

%for simulation
x=time2(600:900);
y=measuredPowerfilt(600:900);
% fs = 3;
% [b,a] = butter(4, .03*2/fs);
% y1 = filtfilt(b,a, y);
y1=y-254;
x1=(x-x(1))';
% x1 = 1:101;
% x1=x1';

[f,g] = fit(x1,y1,'exp1');
timeconstant = 1/f.b;
figure;
plot(f,x1,y1);


figure(7);
hold on;
ax = gca;
ax.LabelFontSizeMultiplier = 2;
ax.TitleFontSizeMultiplier = 2;
title('', 'fontweight', 'bold');
plot(f1,x1_1, y1_1)

plot(time(1470:2900)-time(1470), rpmfilt(1470:2900), 'LineWidth',2); plot(time(1470:2900)-time(1470), bpmfilt(1470:2900), 'LineWidth',2); 
xlabel('Trial time [s]', 'fontweight', 'bold'); ylabel('Cadence [rpm]', 'fontweight', 'bold');
hold off;


%save(fullfile(fp,fn));