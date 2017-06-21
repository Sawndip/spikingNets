function [t, r, z, inp, tgt, score, expvar, e, success] = svm(netPath, mode, d)

dt = 5e-4;

[r, wOut, z, countTrials] = loadData(strcat(netPath, mode));

disp('Loading weights...')

arch = load(strcat(netPath, 'static/arch.dat'));
N = arch(1);
nIn = arch(3);
nOut = arch(4);
G = arch(5);
Q = arch(6);

w0 = load(strcat(netPath, 'static/w0.dat'));
wFb = load(strcat(netPath, 'static/wFb.dat'));
w = G*w0 + Q*wFb*wOut;

%base = load(strcat('/home/neurociencia/svm/base_', code, '.dat'));
%trials = load(strcat('/home/neurociencia/svm/y_geq_x_', mode, '.dat'));

%disp('Loading trials...')
%d = [];

%for i=1:countTrials
%  if (strcmp(code, 'sequential'))
%    d = [d; trials(i,1)*base(:,3)+trials(i,2)*base(:,4), trials(i,3)*base(:,5)];
%  elseif (strcmp(code, 'simultaneous'))
%    d = [d; trials(i,1)*base(:,3), trials(i,2)*base(:,4), trials(i,3)*base(:,5)];
%  end
%end

totalTime = length(d)*dt;
t = (linspace(0, totalTime, length(z)))';

inp = d(1:10:end,3);
tgt = d(1:10:end,4);

disp('Computing PCA...')
[coeff, score, latent, tsquared, expvar, mu] = pca(r');

sz = size(score);

for i=1:sz(2)
  score(:,i) = score(:,i)/max(abs(score(:,i)));
end

disp('Computing eigenvalues...')
e = eig(w);

success = 100*sum((z.*tgt)>0)/sum(abs(tgt)>0);

figure(1)
plot(t, [tgt-2 z-2 inp])
xlabel('Time (s)')
ylim([-3.3, max(inp)+.3])
legend('Target', 'Output', 'PCA1', 'PCA2', 'Input')
