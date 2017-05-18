function [] = doInit(initPath)

rng(42)

N = 2000;
vreset = -65;
vth = -40;
lambda = 4e2;

v = (vth-vreset)*rand(N, 1) + vreset;
r = zeros(N, 1);
h = r;
P = eye(N)/lambda;
wOut = zeros(1, N);

dlmwrite(strcat(initPath, 'v.dat'), v)
dlmwrite(strcat(initPath, 'r.dat'), r)
dlmwrite(strcat(initPath, 'h.dat'), h)
dlmwrite(strcat(initPath, 'P.dat'), P, 'delimiter', ' ')
dlmwrite(strcat(initPath, 'wOut.dat'), wOut, 'delimiter', ' ')
