function [r, wOut, z, w0, wFb, w, e, score, latent, explained] = loadData(path, mode)

currentPath = pwd;
cd(path)
files = dir();

r = [];
wOut = [];
countTrials = 0;

for i=1:length(files)
  
  if length(files(i).name) >= 4
    
    if (strcmp(files(i).name(1), 'r'))
      
      disp(strcat('Loading... ', files(i).name))
      rNow = load(files(i).name);
      r = [r, rNow(:,2:end)]; % first element is repeated
      countTrials = countTrials + 1;

    elseif strcmp(files(i).name(1:4), 'wOut')
      
      wOut = load(files(i).name);
    
    end
  end
end

disp('Loading weights...')
w0 = load('../static/w0.dat');
wFb = load('../static/wFb.dat');
w = .05*w0 + 30*wFb*wOut;

z = (wOut*r)';

d = [];
trials = load(strcat('/home/neurociencia/tmpxor/tmpxor', mode, '.dat'));

disp('Loading trials...')
for i=1:countTrials
  d = [d; load(strcat('/home/neurociencia/tmpxor/tmpxor', num2str(trials(i)), '.dat'))];
end


%% Plot input, target and output %%
if strcmp(mode, 'Train28')
  totalTime = 28;
elseif strcmp(mode, 'Train56')
  totalTime = 56;
elseif (strcmp(mode, 'Train100') || strcmp(mode, 'Tests100'))
  totalTime = countTrials;
elseif strcmp(mode, 'Tests')
  totalTime = 12;
end

timez = linspace(0, 2*totalTime, length(z));
timed = linspace(0, 2*totalTime, length(d));

[coeff, score, latent, tsquared, explained, mu] = pca(r');

figure(1)
plot(timez, z-1, 'r',...
     timez, .5*(score(:,1)/max(score(:,1)))-1, 'b',...
     timed, d(:,2)-1, 'c',...
     timed, d(:,1)+1, 'k')

legend('Output', 'PCA1', 'Target', 'Input')
title('Temporal XOR')
xlabel('Time (s)')
ylim([-2 2])


disp('Computing eigenvalues...')
%% Plot w eigenvalues %%
e = eig(w);

figure(2)
plot(e, 'o')
title('Spectrum of w')
axis('equal')

disp('Computing principal components...')
%% Plot PCAs %%

figure(3)
plot(score(:,1), score(:,2))
xlabel('PCA 1')
ylabel('PCA 2')

cd(currentPath)
