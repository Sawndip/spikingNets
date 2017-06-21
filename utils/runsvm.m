code = 'simultaneous';
basePath = strcat('/home/neurociencia/svm/amplitude/', code, '/');

zz = [];
pcs = [];
expvar = [];
eigenv = [];
s = [];

for i=0:9
  netPath = strcat(basePath, 'net', num2str(i), '/');

  [r, z, inp, tgt, sc, evar, e, success] = svm(netPath, 'test', code);

  zz = [zz z];
  pcs = [pcs sc(:,1:5)];
  expvar = [expvar evar];
  eigenv = [eigenv e];
  s = [s; success];
end
