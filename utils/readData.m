function [r, z, numTrials, trialEnds] = readData(path, format, loadRates)

f = dir(path);

rFiles = regexpi({f.name}, 'r[0-9]+\.dat', 'match');
rFiles = [rFiles{:}];

zFiles = regexpi({f.name}, 'z[0-9]+\.dat', 'match');
zFiles = [zFiles{:}];

if (length(rFiles) > 0)
  numTrials = min(length(rFiles), length(zFiles));
else
  numTrials = length(zFiles);
end

trialEnds = [];
z = [];
r = [];

for i=1:numTrials
  disp(sprintf('Loading... file %d of %d', i, numTrials))
  
  zNow = load(strcat(path, char(zFiles(i))));

  if loadRates
    rNow = load(strcat(path, char(rFiles(i))));
    r = [r rNow];
  end

  if strcmp(format, 'long')
    z = [z zNow];
  elseif strcmp(format, 'short')
    %% -- NEEDS FIXING FOR THE CASE nTargets > 1 -- %%
    sz = size(z);
    aux1 = [zeros(sz(1), max(length(zNow)-sz(2), 0)) z];
    aux2 = [zeros(1, max(sz(2)-length(zNow), 0)) zNow];
    z = [aux1; aux2];
  end
  
  trialEnds = [trialEnds; length(z)];
end

z = z';
r = r';
