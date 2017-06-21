taskPath = '/home/neurociencia/detect/';
filesPath = strcat(taskPath, 'net2/test/');

[r, wOut, z, nt] = readData(filesPath);

z = z';

choice = (max(z(700:end,:)) - mean(z(1:699,:)))';

d = [];

disp('Loading trials...')

for i=100:199
  d = [d; load(strcat(taskPath, 'trials/trial', num2str(i), '.dat'))];
end

inp = d(1:10:end,3);
tgt = d(1:10:end,4);

a = load(strcat(taskPath, 'trials/amplitudes.dat'));
a = a(101:end);

p = linspace(0, 1, 100);
p = flip(p);

fpr = [];
tpr = [];

realNegatives = sum(a==0);
realPositives = sum(a>0);

for i=1:length(p)
  predictedClass = (choice > p(i));
  trueClass = (a > 0);
  fpr = [fpr, sum((predictedClass==1).*(trueClass==0))/realNegatives];
  tpr = [tpr, sum((predictedClass==1).*(trueClass==1))/realNegatives];
end

figure(1)
plot(fpr, tpr, '-b', [0 1], [0 1], '--k')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve')
