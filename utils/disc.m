function [r, z, score, explained, e] = svm(netPath, mode)

[r, wOut, z, countTrials] = loadData(strcat(netPath, mode, '_100/'));

disp('Loading weights...')

w0 = load('../static/w0.dat');
wFb = load('../static/wFb.dat');
w = .04*w0 + 10*wFb*wOut;

trials = load(strcat('../../trials_', mode, '.dat'));

disp('Loading trials...')
d = [];

for i=1:countTrials
  d = [d; load(strcat('../../disc', num2str(trials(i)), '.dat'))];
end

totalTime = length(d)*5e-4;

disp('Computing PCA...')
[coeff, score, latent, tsquared, explained, mu] = pca(r');

disp('Computing eigenvalues...')
e = eig(w);

figure(1)
plot(z)
