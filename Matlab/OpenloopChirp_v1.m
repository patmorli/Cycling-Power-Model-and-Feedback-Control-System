%{
Openloop_v1
Started: Apr. 24, 2018
seperates trials, normalizes to 0 before perturbation and 1 afterwards. 
%}
clear all;
filename = 'Try_inside';


[~, ~, raw] = xlsread([filename  '.xlsx']); 
data = cell2mat(raw(3:end,1));
%logistics = raw(:,2);
%logistics(cellfun(@(logistics) any(isnan(logistics)),logistics)) = [];

%% sort data (1: bpm, 2: time, 3: rpm, 4: power)
p = 0;
t = 1; 
for i = 1:length(data)
    p = p + 1;
    
    if p== 1
        delfreqraw(t) = data(i);
    elseif p == 2
        timeraw(t) = data(i);
    elseif p == 3
        torqueraw(t) = data(i);            %old version, zeroing during
        t = t + 1;
        p = 0;
    end
end


%% get rid of double measurements, maybe we dont need that anymore
timewd = timeraw; %wd..without double
torquewd = torqueraw;
delfreqwd = delfreqraw;

p = 1;
for i = 1:length(timeraw)-1
    if timeraw(i) == timeraw(i+1)
        timewd(p+1) = [];
        torquewd(p+1) = [];
        delfreqwd(p+1) = [];
        p = p - 1;
    end
    p = p + 1;
end


%% get rid of 0s and set time back
torquewd(torquewd == 0) = NaN;
time = timewd - timewd(1);

%% delete all outliers, dont keep nans, delete outliers in time, bpm as well
[torque,torqueidx,torqueoutliers] = deleteoutliers(torquewd, 0.000001, 0);
time(torqueidx) = [];
delfreqwd(torqueidx) = [];

%% calculate rpm and power and bpm
for i = 1:length(time)-1
    rpm(i) = 60000/(time(i+1)-time(i));
    bpm(i) = 60000/(delfreqwd(i) * 2);
    power(i) = torque(i+1) * (2*pi / ((time(i+1) - time(i)) / 1000));
end
time = time(2:end);
delfreq = delfreqwd(2:end);
torque = torque(2:end);

%% normalization
%[bpmtrials, rpmtrials, powertrials, rpmtrialsNorm, bpmtrialsNorm, powertrialsNorm, timetrials] = normalization_datawise(rpm, bpm, time, power);

%% plots 
figure;
yyaxis left; hold on; plot(time, rpm); plot(time, bpm); yyaxis right; plot(time, power);

figure; yyaxis left; hold on; plot(time)



%%
save(filename);
%save('Openloop_Bike_v4_1');