function [r, wOut, z, d, score, explained, e] = svm(netPath, mode)

[r, wOut, z, countTrials] = loadData(strcat(netPath, mode, '_100/'));

disp('Loading weights...')

w0 = load('../static/w0.dat');
wFb = load('../static/wFb.dat');
w = .04*w0 + 10*wFb*wOut;

base = load('../../base.dat');
trials = load(strcat('../../y_geq_x_', mode, '.dat'));

disp('Loading trials...')
d = [];

for i=1:countTrials
  d = [d; trials(i,1)*base(:,2), trials(i,2)*base(:,2), trials(i,3)*base(:,3)];
end

totalTime = length(d)*5e-4;

disp('Computing PCA...')
[coeff, score, latent, tsquared, explained, mu] = pca(r');

disp('Computing eigenvalues...')
e = eig(w);
