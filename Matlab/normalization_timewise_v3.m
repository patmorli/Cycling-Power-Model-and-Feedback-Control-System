%Started: 23March 2018, Patrick Mayerhofer
%normalization_v1 for cutting data
%normalize pilot data to 0 and 1 and for 
%time normalization
clear
close all

%% load your data
 fp = '/Users/patrick/Library/CloudStorage/GoogleDrive-patrick.mayerhofer@locomotionlab.com/My Drive/Cycling Power Control/Work/FeedbackControl/';
 fn = 'Oct190_020_0_012CL_Patv2.mat';
 load(fullfile(fp,fn));
%load('Oct190_020_0_012CLPatv1');


  %% create array that describes index of bpm change
p = 1;
cutindex = 1;
for i = 1:length(targetpower) - 1
    if targetpower(i) ~= targetpower(i+1)
        p = p + 1;
        cutindex(p) = i+1;
    end
end

%  sort data to 30 seconds before perturbation and 30 seconds after %
%  perturbation 


%% organize data for normalization

for i = 1:length(cutindex)-1
    %30 sec indizes
  
    time2 = time(cutindex(i+1));
    %before perturbationtime2
    a = time(time <= time2  - 30);    %all numbers that are 20secs before perturbation
    time1index(i) = find(time == a(end));  %find indizes of the last value, meaning the last one that is 30 secs before
    clear a;
    
    %after perturbation
    a = time(time >= time2 + 59);    %all numbers that are 59 secs after pert
     if a
        time3index = find(time == a(1));    %find indizes of the first value, meaning the first one that is after 30secs
     else
         time3index = length(targetpower);
     end
    %cut data to trials (not normalized yet)
    targetpowertrials{i,1} = targetpower(time1index(i):cutindex(i+1)-1);
    targetpowertrials{i,1} = [targetpowertrials{i,1}', targetpower(cutindex(i+1):time3index)'];
    
    
    bpmtrials{i,1} = bpm(time1index(i):cutindex(i+1)-1);       %before perturbation
    bpmtrials{i,1} = [bpmtrials{i,1}', bpm(cutindex(i+1):time3index)'];         %after perturbation
    
    timetrials{i,1} = time(time1index(i):cutindex(i+1)-1);
    timetrials{i,1} = [timetrials{i,1}', time(cutindex(i+1):time3index)'];
    
    rpmtrials{i,1} = rpm(time1index(i):cutindex(i+1)-1);
    rpmtrials{i,1} = [rpmtrials{i,1}, rpm(cutindex(i+1):time3index)];
    
    powertrials{i,1} = power(time1index(i):cutindex(i+1)-1);
    powertrials{i,1} = [powertrials{i,1}', power(cutindex(i+1):time3index)'];
    
    powerfilttrials{i,1} = powerfilt(time1index(i):cutindex(i+1)-1);
    powerfilttrials{i,1} = [powerfilttrials{i,1}', powerfilt(cutindex(i+1):time3index)'];
    
    aftersteppowertrials{i,1} = powerfilt(cutindex(i+1):time3index);
    aftersteptimetrials{i,1} =time(cutindex(i+1):time3index);
    
    
   
end

for u = 1:length(cutindex)-2
    rpmMean{u,1} = mean(rpm(time1index(u):cutindex(u+1)-1));       %30 secs before perturbation
    rpmMean{u,2} = mean(rpm(time1index(u+1):cutindex(u+2)-1));       %end of trial(30 seconds before NEXT perturbation)
    bpmMean{u,1} = mean(bpm(time1index(u):cutindex(u+1)-1));       
    bpmMean{u,2} = mean(bpm(time1index(u+1):cutindex(u+2)-1));
    powerMean{u,1} = mean(power(time1index(u):cutindex(u+1)-1));       
    powerMean{u,2} = mean(power(time1index(u+1):cutindex(u+2)-1));
    powerfiltMean{u,1} = mean(powerfilt(time1index(u):cutindex(u+1)-1));       
    powerfiltMean{u,2} = mean(powerfilt(time1index(u+1):cutindex(u+2)-1));
    targetpowerMean{u,1} = mean(targetpower(time1index(u):cutindex(u+1)-1));       
    targetpowerMean{u,2} = mean(targetpower(time1index(u+1):cutindex(u+2)-1));

  
        for i = 1:length(bpmtrials{u})
                %zi = (xi - xmeanbeforepert)/(xmeanafterpert-xmeanbeforepert)
                rpmtrialsNorm{u}(i) = (rpmtrials{u}(i) - rpmMean{u,1}) / (rpmMean{u,2} - rpmMean{u,1}); 
                bpmtrialsNorm{u}(i) = (bpmtrials{u}(i) - bpmMean{u,1}) / (bpmMean{u,2} - bpmMean{u,1}); 
                powertrialsNorm{u}(i) = (powertrials{u}(i) - powerMean{u,1}) / (powerMean{u,2} - powerMean{u,1}); 
                powerfilttrialsNorm{u}(i) = (powerfilttrials{u}(i) - powerfiltMean{u,1}) / (powerfiltMean{u,2} - powerfiltMean{u,1}); 
                targetpowertrialsNorm{u}(i) = (targetpowertrials{u}(i) - targetpowerMean{u,1}) / (targetpowerMean{u,2} - targetpowerMean{u,1}); 
        end
        
        for i = 1:length(aftersteppowertrials{u})
             aftersteppowertrialsNorm{u}(i) = (aftersteppowertrials{u}(i) - powerfiltMean{u,1}) / (powerfiltMean{u,2} - powerfiltMean{u,1}); 
        end
        %interpolation to 200 datapoints
        X1=linspace(1,size(aftersteppowertrialsNorm{u},2),200);
        aftersteppowertrialsNormSpline(u,:)=interp1(1:size(aftersteppowertrialsNorm{u},2),aftersteppowertrialsNorm{u},X1,'spline');
        Y1=linspace(1,size(aftersteptimetrials{u},1),200);
        aftersteptimetrialsSpline(u,:)=interp1(1:size(aftersteptimetrials{u},1),aftersteptimetrials{u},Y1,'spline');
        X2=linspace(1,size(powerfilttrialsNorm{u},2),300);
        powerfilttrialsNormSpline(u,:)=interp1(1:size(powerfilttrialsNorm{u},2),powerfilttrialsNorm{u},X2,'spline');
        Y2=linspace(1,size(timetrials{u},2),300);
        timetrialsSpline(u,:)=interp1(1:size(timetrials{u},2),timetrials{u},Y2,'spline');
        X3=linspace(1,size(targetpowertrialsNorm{u},2),300);
        targetpowertrialsNormSpline(u,:)=interp1(1:size(targetpowertrialsNorm{u},2),targetpowertrialsNorm{u},X3,'spline');
        X4 = linspace(1,size(powerfilttrials{u},2),300);
        powerfilttrialsSpline(u,:)=interp1(1:size(powerfilttrials{u},2),powerfilttrials{u},X3,'spline');


        
end


save(fullfile(fp,fn));