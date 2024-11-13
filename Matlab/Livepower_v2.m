%Livepower_v2
%Started 14. May, 2018
%shows power in liveplot

clear all
clc

targetpower = 156;
ytop = 300;
ybottom = 0;
upperline = targetpower*1.05;
lowerline = targetpower*0.95;
formatSpec = '%f %d';

%% Setup serial communication
try fclose(instrfind); delete(S), clear S; catch, end

S = serial('/dev/cu.usbmodem14111','BaudRate',115200);
S.InputBufferSize = 1000000; % read only one byte every time
 
try
    fopen(S);
catch err
    fclose(instrfind);
    error('Make sure you select the correct COM Port where the Arduino is connected.');
end
flushinput(S); 
%% read data
tic
i = 1;
h1= figure(1); clf; hold on
%figure(2); clf; hold on
while toc < 60
while toc < 60 && S.bytesavailable
    toc
    y(i)=fscanf(S, formatSpec);
    
    if y(i) < 300
        figure(1); plot(toc,y(i), 'ro'); hold on; xlim([toc-10 toc+10]); ylim([ybottom ytop]);
        refline(0, upperline);
        refline(0, lowerline);
    end
    %if i>4, figure(1); plot([toc(i-4) toc(i-2)], [y(i-4) y(i-2)],'r-'); end
    i = i + 1;
end
end

%end
% angVel =
fclose(S);