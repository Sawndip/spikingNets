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
  
  // LIF Parameters
  float vth = -40.0;
  float vreset = -65.0;
  float vinf = -39.0;
  float tref = .002; //refractory time
  float tm = .01; //membrane relaxation
  float td = .03; //decrease
  float tr = .002; //rise

  // Network parameters
  int N = 2000; //neurons
  float p = .1; //sparsity
  int nIn = 2; //inputs
  int nOut = 1; //outputs
  float G = .04;
  float Q = 10.0;

  // Initial values
  mat w, wOut;
  w.load("/home/neurociencia/svm/test100/w.dat", raw_ascii);
  wOut.load("/home/neurociencia/svm/test100/wOut.dat", raw_ascii);

  mat wIn = 2.0*arma::randu<mat>(N, nIn) - 1.0*arma::ones<mat>(N, nIn);
  mat wFb = 2.0*arma::randu<mat>(N, nOut) - 1.0*arma::ones<mat>(N, nOut);
  vec v = (vth - vreset)*arma::randu<vec>(N) + vreset; // uniformly distributed within the linear regime
  vec r = arma::zeros<vec>(N);
  vec h = arma::zeros<vec>(N);

  _Net myNet = createSpikeNet(vth, vreset, vinf, tref, tm, td, tr, N, p, nIn, nOut, G, Q, w, wIn, wOut, wFb, v, r, h);

  // Read input and target from file
  mat svmData;
  svmData.load("/home/neurociencia/svm/y_geq_1x_0_5e5_n200.dat", raw_ascii);
  printf("%i %i \n", svmData.n_rows, svmData.n_cols);
  mat inp = svmData.cols(0,1).t();
  mat tgt = svmData.col(3).t();
   
  // Integration parameters
  int T = svmData.n_rows;
  float dt = 5e-5;
  float totalTime = T*dt; //total time in seconds

  // True only for checking chaotic behaviour by removing one spike
  bool spikeTest = false;

  // FORCE parameters
  int trainStep = 5;
  float trainRate = 4e5; //the smaller the faster the training occurs
  int trainStart = 0; //start training after trainStart*dt time
  int trainStop = T/2; //stop training here but keep net running
  int saveRate = 100; //keep Nyquist-Shannon in mind when choosing this

  myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, trainStart, trainStop, trainRate, saveRate, spikeTest);

  inp = inp.t();
  inp.save("/home/neurociencia/svm/inp.dat", raw_ascii);

  wOut = myNet.wOut;
  wOut.save("/home/neurociencia/svm/wOut.dat", raw_ascii);

  w = myNet.w;
  w.save("/home/neurociencia/svm/w.dat", raw_ascii);

  tgt = tgt.t();
  tgt.save("/home/neurociencia/svm/tgt.dat", raw_ascii);
  
  return 0;
}
