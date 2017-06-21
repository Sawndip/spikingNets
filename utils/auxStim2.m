function [s1 s2] = auxStim2(inp)

s1 = inp(1);
s2 = inp(1);

stimStart = 0;

for i=2:length(inp)
  tmp = inp(i)-inp(i-1);

  if tmp==1
    stimStart = stimStart + 1;
  end

  s1 = [s1 inp(i).*mod(stimStart, 2)];
  s2 = [s2 inp(i).*mod(stimStart+1, 2)];
end
