function [r, wOut, z, w0, wFb, w, e, score, explained] = loadData(path, mode)

currentPath = pwd;
cd(path)
files = dir();

r = [];
wOut = [];

for i=1:length(dir)
  
  if length(files(i).name) >= 4
    
    if strcmp(files(i).name(1), 'r')
      
      disp(strcat('Loading... ', files(i).name))
      r = [r, load(files(i).name)];
    
    elseif strcmp(files(i).name(1:4), 'wOut')
      
      wOut = load(files(i).name);
    
    end
  end
end

disp('Loading weights...')
w0 = load('../init/w0.dat');
wFb = load('../init/wFb.dat');
w = .05*w0 + 20*wFb*wOut;

z = (wOut*r)';

d = [];
trials = load(strcat('/home/neurociencia/tmpxor/tmpxor', mode, '.dat'));

disp('Loading trials...')
for i=1:length(trials)
  d = [d; load(strcat('/home/neurociencia/tmpxor/tmpxor', num2str(trials(i)), '.dat'))];
end


%% Plot input, target and output %%
if strcmp(mode, 'Train28')
  totalTime = 28;
elseif strcmp(mode, 'Train56')
  totalTime = 56;
elseif strcmp(mode, 'Test')
  totalTime = 12;
end

timez = linspace(0, 2*totalTime, length(z));
timed = linspace(0, 2*totalTime, length(d));

figure(1)
plot(timez, z-1, 'r', timed, d(:,2)-1, 'b', timed, d(:,1)+1, 'k')
legend('Output', 'Target', 'Input')
title('Temporal XOR')
xlabel('Time (s)')
ylim([-2 2])


disp('Computing eigenvalues...')
%% Plot w eigenvalues %%
e = eig(w);

figure(2)
plot(e, 'o')
title('Spectrum of w')


disp('Computing principal components...')
%% Plot PCAs %%
[coeff, score, latent, tsquared, explained, mu] = pca(r');

figure(3)
plot(score(:,1), score(:,2))
xlabel('PCA 1')
ylabel('PCA 2')

cd(currentPath)
