close all; clear all; clc

dt = 5e-4;
netPath = '/home/neurociencia/tmpxor/net40/';
mode = 'test';

[r, wOut, z, ct] = loadData(strcat(netPath, mode, '/'));

cd(netPath)

w0 = load('static/w0.dat');
wFb = load('static/wFb.dat');
arch = load('static/arch.dat');
G = arch(5);
Q = arch(6);

w = G*w0 + Q*wFb*wOut;

disp('Calculating eigenvalues...')
e = eig(w);

disp('Loading trials...')
d = [];

for i=0:min(99, ct)
  if strcmp(mode, 'train')
    k = i;
  elseif strcmp(mode, 'test')
    k = i+100;
  end

  d = [d; load(strcat('trials/trial', num2str(k), '.dat'))];

end

inp = d(1:10:end, 3);
tgt = d(1:10:end, 4);
t = linspace(0, length(d)*dt, length(tgt));

% PCAs %
disp('Computing PCs...')
[coeff, sc, latent, tsquared, expvar, mu] = pca(r');

for i=1:2000
  sc(:,i) = sc(:,i)/max(abs(sc(:,i)));
end

% %plot stuff %

tend = min(length(z), length(inp));
z = z(1:tend);
inp = inp(1:tend);
tgt = tgt(1:tend);
t = t(1:tend);

figure(1)
plot(t, tgt, t, z)
xlabel('Time (s)')
legend('Target', 'Output')

figure(2)
plot(t, [inp + 1, tgt - 1, sc(:,1)-1, sc(:,2)+1])
xlabel('Time (s)')
legend('Input', 'Target', 'PCA1', 'PCA2')
ylim([-2.1 2.1])

cd('~/spikingNets/utils')

% Check learning %

% naive, biased way
check = (z.*tgt) > 0;
success = 100*sum(check)/sum(abs(tgt)>0);
disp('Success:')
disp(success)

% proper ROC curve analysis - bundling hits/misses together as true positives,
% since we are only concerned with the S/N ratio here
% So, the "real" performance of the classifier is defined above, but the ROC
% curve tells us whether the network does/doesn't produce spurious signals when
% no input is presented

p = linspace(0, 1, 100); % ROC curve parameter
fpr = [];
tpr = [];

for i=1:length(p)
  predictedClass = abs(z(1:tend)) > p(i);
  trueClass = abs(tgt(1:tend)) > 0;
  realNegatives = sum(trueClass == 0);
  realPositives = sum(trueClass == 1);
  fpr = [fpr, sum((predictedClass == 1).*(trueClass == 0))/realNegatives];
  tpr = [tpr, sum((predictedClass == 1).*(trueClass == 1))/realPositives];
end

figure(3)
plot(fpr, tpr, 'b', [0 1], [0 1], '--k')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve')
