%Cadencebase_v1
%Started 11. April, 2018
%shows power in liveplot

clear all
clc
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
 
%% read data
tic
i = 1;
h1= figure(1); clf; hold on
figure(2); clf; hold on
while toc < 40
    toc
    y(i)=fscanf(S, formatSpec);
    i = i + 1;
    if mod(i,2)==1, figure(1); plot(toc,y(i-1), 'ro'); hold on; xlim([toc-10 toc+10])
    else, figure(2); plot(toc,y(i-1),'bo'); hold on; xlim([toc-10 toc+10])
    end
    
    %if i>4, figure(1); plot([toc(i-4) toc(i-2)], [y(i-4) y(i-2)],'r-'); end
end

% angVel =
fclose(S);