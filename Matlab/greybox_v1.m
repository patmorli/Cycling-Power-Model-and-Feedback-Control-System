%% grey tank stuff
clear all
%% matlab example
load(fullfile(matlabroot,'toolbox','ident','iddemos','data','twotankdata'));
z = iddata(y,u,0.2,'Name','Two tanks');

FileName = 'twotanks_c';

Order = [1 1 2];

Parameters = {0.5;0.0035;0.019; ...
    9.81;0.25;0.016};

InitialStates = [0;0.1];

Ts = 0;

nlgr = idnlgrey(FileName,Order,Parameters,InitialStates,Ts, ...
    'Name','Two tanks');

nlgr.Parameters(1).Fixed = true;
nlgr.Parameters(4).Fixed = true;
nlgr.Parameters(5).Fixed = true;


nlgr = nlgreyest(z,nlgr);

opt = nlgreyestOptions;
opt.Display = 'on';
opt.SearchOptions.MaxIterations = 50;


%%
clear all
opt = nlgreyestOptions;
opt.Display = 'on';
opt.SearchOptions.MaxIterations = 50;

load(fullfile(matlabroot,'toolbox','ident','iddemos','data','dcmotordata'));
z = iddata(y,u,0.1,'Name','DC-motor');

file_name = 'dcmotor_m';
Order = [2 1 2];
Parameters = [1;0.28];
InitialStates = [0;0];

init_sys = idnlgrey(file_name,Order,Parameters,InitialStates,0, ...
    'Name','DC-motor');

sys = nlgreyest(z,init_sys,opt);