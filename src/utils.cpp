#define INFINITY
#include "mconvert.h"
#include "createSpikeNet.h"
#include "runSpikeNet.h"
#include <typeinfo>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <armadillo>
using namespace std;
using namespace arma;

mat uhlenbeck(float theta, float mu, float sigma, float dt, int T)
{
  vec t = arma::linspace<vec>(0, T*dt, T);
  mat x(1, T); x.fill(mu);

  for (int i = 1; i < T; i++)
  {
    x(i) = theta*(mu - x(i-1))*dt + sigma*randn();
  }

  return x;
}


int main(){
  
  arma_rng::set_seed(42);

  // LIF Parameters
  float vth = -40.0;
  float vreset = -65.0;
  float vinf = -39.0;
  float tref = .002; //refractory time
  float tm = .01; //membrane relaxation
  float td = .03; //decrease
  float tr = .002; //rise

  // Network parameters
  int N = 500; //neurons
  float p = .1; //sparsity
  int nIn = 1; //inputs
  int nOut = 1; //outputs
  float G = 0.04;
  float Q = 10.0;

  // Initial values
  sp_mat sp_w = sprandn(N, N, p)/sqrt(N*p);
  mat w(sp_w);
  mat wIn = 2.0*arma::randu<mat>(N, nIn) - 1.0*arma::ones<mat>(N, nIn);
  mat wOut = arma::zeros<mat>(nOut, N);
  mat wFb = 2.0*arma::randu<mat>(N, nOut) - 1.0*arma::ones<mat>(N, nOut);
  vec v = (vth - vreset)*arma::randu<vec>(N) + vreset; // uniformly distributed within the linear regime
  vec r = arma::zeros<vec>(N);
  vec h = arma::zeros<vec>(N);

  _Net myNet = createSpikeNet(vth, vreset, vinf, tref, tm, td, tr, N, p, nIn, nOut, G, Q, w, wIn, wOut, wFb, v, r, h);
  
  // Integration parameters
  float totalTime = 1; //total time in seconds
  float dt = 5e-5; //integation timestep
  int T = round(totalTime/dt); //total time in units of dt
  printf("%i\n", T);
  // True only for checking chaotic behaviour by removing one spike
  bool spikeTest = false;

  // FORCE parameters
  int trainStep = 5;
  float trainRate = 400; //the smaller the faster the training occurs
  int trainStart = (int)T/4; //start training after trainStart*dt time
  int trainStop = (int)3*T/4; //stop training here but keep net running
  int saveRate = 10; //keep Nyquist-Shannon in mind when choosing this
  int saveFORCE = 2000;
  // Input and target
  mat time = dt*linspace<vec>(0, T, T);
  mat inp = 0.0*uhlenbeck(1.0, 0.0, 1.0, dt, T);
  mat tgt = arma::ones<mat>(nOut, T);
  
  runSpikeNet(myNet, inp, tgt, dt, trainStep, trainStart, trainStop, trainRate, saveRate, saveFORCE, spikeTest);
  
  return 0;
}