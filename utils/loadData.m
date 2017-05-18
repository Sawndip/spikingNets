function [r, wOut, z, countTrials] = loadData(path)

currentPath = pwd;
cd(strcat(path))
files = dir();

r = [];
wOut = [];
countTrials = 0;

for i=1:length(files)
  
  if length(files(i).name) >= 4

    if (strcmp(files(i).name(1), 'r') && countTrials < 1000)
        disp(strcat('Loading... ', files(i).name))
        rNow = load(files(i).name);
        r = [r, rNow];
        countTrials = countTrials + 1;
    elseif strcmp(files(i).name(1:4), 'wOut')
      
      wOut = load(files(i).name);
    
    end
  end
end

%disp('Loading weights...')
%w0 = load('../static/w0.dat');
%wFb = load('../static/wFb.dat');
%w = .04*w0 + 10*wFb*wOut;

z = (wOut*r)';

%d = [];
%trials = load(strcat(path, trialsFile));
%base = load(strcat(path, 'base.dat'));

%disp('Loading trials...')
%for i=1:countTrials
%  d = [d; trials(i,1)*base(:,2), trials(i,2)*base(:,2), base(:,3)];
%end

%totalTime = length(d)*dt;

%% -- REALLY HARDCODED -- %%
%d = d(1:10:end, :);
%idx = 1:length(d);
%idx = idx(mod(idx, 400) ~= 0);
%d = d(idx, :);
%% -- You are now leaving the really hardcoded sector -- %%

%timez = linspace(0, totalTime, length(z));
%timed = linspace(0, totalTime, length(d));

%if (length(z) == length(d))
%  tgt = d(:,2)-1;
%  signal = z.*(tgt ~= 0);
%  test = sign(signal).*sign(tgt);
%  test = test(test ~= 0);
%  disp('Success rate: ')
%  disp(sum(test)/length(test))
%else
%  disp('SAMPLING ERROR')
%  disp(strcat('length(z) = ', length(z)))
%  disp(strcat('length(d) = ', length(d)))
%  test = 0;
%end

%disp('Computing principal components...')
%[coeff, score, latent, tsquared, explained, mu] = pca(r');

%figure(1)
%plot(timez, z, 'r',...
%     timed, d(:,2), 'c')

%legend('Output', 'Target')
%title('Temporal XOR')
%xlabel('Time (s)')

%disp('Computing eigenvalues...')
%% Plot w eigenvalues %%
%e = eig(w);

%figure(2)
%plot(e, 'o')
%title('Spectrum of w')
%axis('equal')

%% Plot PCAs %%

%figure(3)
%plot(score(:,1), score(:,2))
%xlabel('PCA 1')
%ylabel('PCA 2')

%cd(currentPath)
