close all; clear all; clc

dt = 5e-4;
nTrialsTrain = 200;
net = 1;
netPath = strcat('/home/neurociencia/tmpxor/net', num2str(net), '/');
mode = 'test';

[r, z, ct, te] = readData(strcat(netPath, mode, '/'), 'long', false);

cd(netPath)

wOut = load('dyn/wOut.dat');
w0 = load('static/w0.dat');
wFb = load('static/wFb.dat');
arch = load('static/arch.dat');
G = arch(5);
Q = arch(6);

cd('..')

disp('Loading trials...')
d = [];

for i=0:min(99, ct)
  if strcmp(mode, 'train')
    k = i;
  elseif strcmp(mode, 'test')
    k = i + nTrialsTrain;
  end

  d = [d; load(strcat('trials', num2str(net), '/trial', num2str(k), '.dat'))];

end

inp = d(:, 3);
tgt = d(:, 4:end);
t = linspace(0, length(d)*dt, length(tgt));

% Calculate PCs %
disp('Computing PCs...')

[cf sc lat tsq evar mu] = pca(r);

% %plot stuff %

tend = min(length(z), length(inp));
z = z(1:tend, :);
inp = inp(1:tend, :);
tgt = tgt(1:tend, :);
t = t(1:tend);

figure(1)
plot(t, tgt, t, z)
xlabel('Time (s)')
legend('Target', 'Output')

cd('~/spikingNets/utils')

% Check learning %

% naive, biased way
check = (z.*tgt) > 0;
success = 100*sum(check)./sum(abs(tgt)>0);
disp(sprintf('Motor success: %d', success(1)))
if length(success) > 1
  disp(sprintf('Decision success: %d', success(2)))
end

% proper ROC curve analysis - bundling hits/misses together as true positives,
% since we are only concerned with the S/N ratio here
% So, the "real" performance of the classifier is defined above, but the ROC
% curve tells us whether the network does/doesn't produce spurious signals when
% no input is presented

p = linspace(0, 1, 100); % ROC curve parameter
fpr = [];
tpr = [];

for i=1:length(p)
  predictedClass = abs(z(1:tend, 1)) > p(i);
  trueClass = abs(tgt(1:tend, 1)) > 0;
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

AUC = -trapz(fpr, tpr);
disp(sprintf('AUC: %d', AUC))
