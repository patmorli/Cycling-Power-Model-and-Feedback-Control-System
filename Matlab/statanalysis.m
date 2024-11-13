function [meanpercentagepacingerror, stdpercentagepacingerror, cv] = statanalysis(fp,fn)
%statistical analysis, Patrick Mayerhofer, October, 2019

load(fullfile(fp,fn));


%pacing error
for i=1:size(targetpowertrials,1)-1
    
%     meanpacingerror(i)=mean(abs(targetpower(time1index(i):cutindex(i+1)-1)-...
%         powerfilt(time1index(i):cutindex(i+1)-1)));
%     stdpacingerror(i)=std(abs(targetpower(time1index(i):cutindex(i+1)-1)-...
%         powerfilt(time1index(i):cutindex(i+1)-1)));
%     absolutepacingerror(i) = sum((abs(targetpower(time1index(i):cutindex(i+1)-1)-...
%         powerfilt(time1index(i):cutindex(i+1)-1))));

    meanpercentagepacingerror(i) = mean((abs(targetpower(time1index(i):cutindex(i+1)-1)-...
        powerfilt(time1index(i):cutindex(i+1)-1)))./targetpower(time1index(i):cutindex(i+1)-1));
    stdpercentagepacingerror(i) = std((abs(targetpower(time1index(i):cutindex(i+1)-1)-...
        powerfilt(time1index(i):cutindex(i+1)-1)))./targetpower(time1index(i):cutindex(i+1)-1));
    cv(i)=std(powerfilt(time1index(i):cutindex(i+1)-1))./mean(powerfilt(time1index(i):cutindex(i+1)-1));
  %cv(i)=std(Pfilt(time1index(i):cutindex(i+1)-1))./mean(Pfilt(time1index(i):cutindex(i+1)-1));
end


% X1=linspace(1,size(zyklus,1),100);
%     zyklus=interp1(1:size(zyklus,1),zyklus,X1,'spline');
end

