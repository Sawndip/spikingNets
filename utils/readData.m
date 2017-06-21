function [r, wOut, z, numTrials] = readData(path)

f = dir(path);

rFiles = regexpi({f.name}, 'r[0-9]+\.dat', 'match');
rFiles = [rFiles{:}];
numTrials = length(rFiles);

wOutFiles = regexpi({f.name}, 'wOut[0-9]+\.dat', 'match');
wOutFiles = [wOutFiles{:}];
wOut = load(strcat(path, char(wOutFiles(end))));
wOut=load('/home/neurociencia/detect/net2/train/wOut0099.dat');
z = [];
r = [];

for i=1:numTrials
  disp(sprintf('Loading... file %d of %d', i, numTrials))
  rNow = load(strcat(path, char(rFiles(i))));
  zNow = wOut*rNow;
  
  sz = size(z);
  aux1 = [zeros(sz(1), max(length(zNow)-sz(2), 0)) z];
  aux2 = [zeros(1, max(sz(2)-length(zNow), 0)) zNow];

  z = [aux1; aux2];
  r = [r rNow];
end
