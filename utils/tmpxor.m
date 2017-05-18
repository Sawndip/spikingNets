netPath = '/home/neurociencia/tmpxor/depasquale/net4/';

[r, wOut, z, ct] = loadData(strcat(netPath, 'test_100/'));

cd(netPath)

w0 = load('static/w0.dat');
wFb = load('static/wFb.dat');
arch = load('static/arch.dat');
G = arch(5);
Q = arch(6);

w = G*w0 + Q*wFb*wOut;

disp('Calculating eigenvalues...')
e = eig(w);

cd('..')

trials = load('test_trials.dat');
d = [];

disp('Loading trials...')

for i=1:length(trials)
  d = [d; load(strcat('trial', num2str(trials(i)), '.dat'))];
end

tgt = d(1:10:end,2);

% Test learning %

check = z.*((abs(z) > .5).*sign(tgt)) > 0;
t = linspace(0, 100, length(check));

% The number of vertical stripes is the number of correct classifications
%
plot(t, check)
