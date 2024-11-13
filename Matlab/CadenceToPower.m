t = time';

fbracket = 39;
rbracket = 17;
GR = fbracket/rbracket;

rw=0.3; %radius wheel
lca = 0.17; %length crank arm

% calculated speed from gear ratio and rpm
v = (((rw * 2*pi*GR.*rpm)/60))';

%% derive
dv = nanmean([diff([v NaN]); diff([NaN v])]); % central difference
dt = nanmean([diff([t NaN]); diff([NaN t])]);
vdot = dv./dt;

Pnewopt = (rw/lca)*GR*m*vdot.*v + (rw/lca)*GR*c1*v.^3; 