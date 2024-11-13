%Started: 23March 2018, Patrick Mayerhofer
%normalization_v1 for cutting data
%normalize pilot data to 0 and 1 and for 
%time normalization

  %% create array that describes index of bpm change
p = 1;
cutindex = 1;
for i = 1:length(bpm) - 1
    if bpm(i) ~= bpm(i+1)
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
    %before perturbation
    a = time(time <= time2 - 30000);    %all numbers that are 30secs before perturbation
    time1index = find(time == a(end));  %find indizes of the last value, meaning the last one that is 30 secs before
    clear a;
    
    %after perturbation
    a = time(time >= time2 + 30000);    %all numbers that are 30 secs after pert
    time3index = find(time == a(1));    %find indizes of the first value, meaning the first one that is after 30secs
    
    %cut data to trials (not normalized yet)
    bpmtrials{i,1} = bpm(time1index:cutindex(i+1)-1);       %before perturbation
    bpmtrials{i,2} = bpm(cutindex(i+1):time3index);         %after perturbation
    
    timetrials{i,1} = time(time1index:cutindex(i+1)-1);
    timetrials{i,2} = time(cutindex(i+1):time3index);
    
    rpmtrials{i,1} = rpm(time1index:cutindex(i+1)-1);
    rpmtrials{i,2} = rpm(cutindex(i+1):time3index);
    
    powertrials{i,1} = power(time1index:cutindex(i+1)-1);
    powertrials{i,2} = power(cutindex(i+1):time3index);
end

for u = 1:length(cutindex)-2
    rpmMean{u,1} = mean(rpmtrials{u,1});       %30 secs before perturbation
    rpmMean{u,2} = mean(rpmtrials{u+1,1});       %end of trial(30 seconds before NEXT perturbation)
    bpmMean{u,1} = mean(bpmtrials{u,1});       
    bpmMean{u,2} = mean(bpmtrials{u+1,1});
    powerMean{u,1} = mean(powertrials{u,1});       
    powerMean{u,2} = mean(powertrials{u+1,1});

    for p = 1:2
        for i = 1:length(bpmtrials{u,p})
                %zi = (xi - xmeanbeforepert)/(xmeanafterpert-xmeanbeforepert)
                rpmtrialsNorm{u,p}(i) = (rpmtrials{u,p}(i) - rpmMean{u,1}) / (rpmMean{u,2} - rpmMean{u,1}); 
                bpmtrialsNorm{u,p}(i) = (bpmtrials{u,p}(i) - bpmMean{u,1}) / (bpmMean{u,2} - bpmMean{u,1}); 
                powertrialsNorm{u,p}(i) = (powertrials{u,p}(i) - powerMean{u,1}) / (powerMean{u,2} - powerMean{u,1}); 
        end

    end
end