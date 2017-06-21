% Implementation of https://arxiv.org/pdf/physics/0612169.pdf - page 7

% Input: NxN adjacency matrix of directed, weighted graph

% Output: tau (resp t) is a Nx4 vector containing the (maximum
% possible) number of cycles, middleman, in and out for each node
% Therefore, the clustering coefficients are c = tau./t

function [tau, t] = clusters(w)

sz = size(w);

if (sz(1)==sz(2))
  N = sz(1);
else
  disp('Matrix is not square!')
  return
end

w = abs(w - diag(w)*ones(1,N).*eye(N)); % discard self connections & weight sign

w3 = nthroot(w, 3);
w0 = w ~= 0;

din = (sum(w0'))';
dout = (sum(w0))';
dboth = diag(w0^2);

tau = [diag(w3^3), diag(w3*w3'*w3), diag(w3'*w3*w3), diag(w3*w3*w3')];
t = [din.*dout-dboth, din.*dout-dboth, din.*(din-1), dout.*(dout-1)];
