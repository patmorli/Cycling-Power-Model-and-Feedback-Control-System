%% Subject
sub_m = 60.66;
sub_l = 0.90;

%% general model for normalization
m = 70;
g = 9.81;
l = 0.97;
P = 100;
cadence = 75;

%% Power
P_ratio = P/(m*g^0.5*l^1.5);

new_Power = P_ratio * sub_m * g^0.5 * sub_l^1.5;

%% Cadence
f = cadence/60;
f_ratio = f/(sqrt(g*l));

new_f = f_ratio * (sqrt(g*sub_l));
new_cadence = new_f * 60; 

