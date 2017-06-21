function [r, wOut, z, countTrials] = loadData(path)

currentPath = pwd;
cd(path)
files = dir();

r = [];
wOut = [];
countTrials = 0;

for i=1:length(dir)
  
  if length(files(i).name) >= 4

    if strcmp(files(i).name(1), 'r')
        disp(strcat('Loading... ', files(i).name))
       
        rNow = load(files(i).name);
        r = [r, rNow];
        countTrials = countTrials + 1;
    elseif strcmp(files(i).name(1:5), 'wOut0')
      
      wOut = load(files(i).name);
    
    end
  end
end

z = (wOut*r)';

cd(currentPath)
