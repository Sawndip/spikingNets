% Check if the P matrix in the FORCE algorithm does in fact approximate the inverse of the (regularized, not centered) covariance matrix
%
% i.e. inv(P)_ij = sum_t (r_i*r_j) + lambda*Id_ij 

function [r, P, avgError] = pTest(r, lambda, varargin)

  if nargin == 0
  
    rng(42)

    N = 100; % number of coordinates
    T = 1e4; % duration of simulation

    lambda = 4e5; % learning rate
    rMean = 25; % mean firing rate

    t = linspace(0, 10, T);

    phase = 2*pi*rand(1, N); % random phase
    noise = .05*randn(T, N); % random noise

    signal = (sin(2*pi*t))'*cos(phase) + (cos(2*pi*t))'*sin(phase);

    r = rMean + signal + noise; % same signal + != phase + noise
  
  elseif nargin > 0
    
    [T, N] = size(r);
  
  end
  
  P = eye(N)/lambda;

  avgError = [];

  for i=1:T

    disp(strcat('Iteration ', num2str(i), ' of ', num2str(T)))

    rSoFar = r(1:i,:);
    rNow = r(i,:);

    covPlusMeanNow = rSoFar'*rSoFar;
    targetNow = covPlusMeanNow + lambda*eye(N); % should = inv(P)

    P = P - (P*rNow'*rNow*P)/(1 + rNow*P*rNow');

    avgError = [avgError; mean(mean(abs(inv(P) - targetNow)))];

  end
