%{
Baselinecheck_v1
Started: Apr. 9, 2018
Calculates mean bpm, deletes outliers
%}
clear all;

filename = 'Baselinecheck_Bike_v2'


data = xlsread([filename  '.xlsx']); 

%% sort data (1: time, 2: torque)
p = 0;
t = 1; 
for i = 1:length(data)
    p = p + 1;
    
    if p == 1
        time1(t) = data(i);
    elseif p == 2
        rpm1(t) = data(i);
    else
        power1(t) = data(i);
        t = t + 1;
        p = 0;
    end
end


%% get rid of double measurements
timewd = time1; %wd..without double
powerwd = power1;
rpmwd = rpm1;
p = 1;
for i = 1:length(time1)-1
    if time1(i) == time1(i+1)
        timewd(p+1) = [];
        powerwd(p+1) = [];
        rpmwd(p+1) = [];
        p = p - 1;
    end
    p = p + 1;
end


%% get rid of 0s and set time back
rpmwd(rpmwd == 0) = NaN;
powerwd(powerwd == 0) = NaN;
time = timewd - timewd(1);

%% delete all outliers, keep nans, calculate mean and std of last 30 seconds
[rpm,rpmidx,poweroutliers] = deleteoutliers(rpmwd, 0.05, 1);
[power,poweridx,rpmoutliers] = deleteoutliers(powerwd, 0.05, 1);

rpmMean = nanmean(rpm(time(end)-time<=180000));
rpmStd = nanstd(rpm(time(end)-time<=180000));

powerMean = nanmean(power(time(end)-time<=180000));
powerStd = nanstd(power(time(end)-time<=180000));

plot(time, powerwd, 'r');
hold on;
axis([0 inf 60 160])
plot(time, power, 'b');
a = find(time(end)-time<=180000);
line([time(a(1)), time(a(end))], [powerMean, powerMean], 'Color', 'blue', 'LineWidth', 3, 'LineStyle', '--');

save(filename);



