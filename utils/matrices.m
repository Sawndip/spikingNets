% Some analysis of the initial (G*w0) and effective (G*w0, Q*wFb*wOut)
% synaptic matrices in order to see how training affects:
% 
% 1. Weight distribution: normal by construction pre-training, how does it
% change after FORCE? Laje et al. (2013) report the post-training distribution
% has mean = 0, but greater sd and kurtosis.
%
% 2. Spectrum: the spectrum of a matrix with NxN entries drawn iid from
% a N(0,1) distribution is uniformly distributed in the unit disc as N->
% \infty. This seems to be the case for w0 too, even though it is sparse
% (p=.1) - with the disc scaling ~1/sd*2. The spectral radius is relevant for
% stability issues; typically, rho < 1 implies echo state property, though not
% always.
%
% 3. Eigenspaces and their relation to principal components: wFb*wOut is rank
% 1 and has kernel = orthogonal complement of wOut, so that each
% neuron is fed back the projection of the neural trajectory onto wOut
% times the feedback weight wFb_i.
%
% 4. Clustering coefficients: are they reminiscent of other architectures, such
% as LSTM?

netPath = '/home/neurociencia/tmpxor/randtimes/net7/';

arch = load(strcat(netPath, 'static/arch.dat'));
N = arch(1);
p = arch(2);
G = arch(5);
Q = arch(6);
lambda = arch(7);

w0 = load(strcat(netPath, 'static/w0.dat'));
wFb = load(strcat(netPath, 'static/wFb.dat'));
wOut = load(strcat(netPath, 'init/wOut.dat'));

w = G*w0 + Q*wFb*wOut;

%s = linspace(0, 1, 100);
%e = [];

%for i=1:length(s)
%  disp(strcat('Iteration ', num2str(i), ' of ', num2str(length(s))))
%  e = [e, eig(G*(1-s(i))*w0 + Q*s(i)*wFb*wOut)];
%end

% Clustering

[tau0, t0] = clusters(G*w0);
[tau, t] = clusters(w);

c0 = tau0./t;
c = tau./t;

h0_cyclic = histogram(c0(:,1));
x0 = h0_cyclic.BinEdges(2:end) - h0_cyclic.BinWidth/2;
y0 = h0_cyclic.Values;

h_cyclic = histogram(c(:,1));
x = h_cyclic.BinEdges(2:end) - h_cyclic.BinWidth/2;
y = h_cyclic.Values;

plot(x0, y0, 'b', x, y, 'r')
