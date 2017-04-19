#define INFINITY
#include <typeinfo>
#include "mconvert.h"
#include "createSpikeNet.h"
#include "runSpikeNet.h"
#include "stimulus.h"
#include <armadillo>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <sstream>
using namespace std;
using namespace arma;

int main()
{
  arma_rng::set_seed(42);
  wall_clock timer;

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
  int nOut = 2; //outputs
  float G = .05;
  float Q = 20.0;

  // Initial values
  sp_mat sp_w = sprandn(N, N, p)/sqrt(N*p); //random initial weights
  mat w(sp_w);
  mat wIn = 2.0*arma::randu<mat>(N, nIn) - 1.0*arma::ones<mat>(N, nIn); //uniform [-1,1]
  mat wOut = arma::zeros<mat>(nOut, N); //zeros
  mat wFb = 2.0*arma::randu<mat>(N, nOut) - 1.0*arma::ones<mat>(N, nOut); //uniform [-1,1]
  vec v = (vth-vreset)*arma::randu<vec>(N)+vreset;
  vec r = arma::zeros<vec>(N);
  vec h = arma::randu<vec>(N);
  
  // Initialize net
  _Net myNet = createSpikeNet(vth, vreset, vinf, tref, tm, td, tr, N, p, nIn, nOut, G, Q, w, wIn, wOut, wFb, v, r, h);
 
  //Integration parameters
  float totalTime = 7; //total time in seconds
  float dt = 1e-1;
  int T = (int)totalTime/dt; //total time in units of dt
  int K = 70; //length of vector r(t)
  int saveRate = round(T/K); //sampling rate of r

  // True only for checking chaotic behaviour by removing one spike
  bool spikeTest = false;
 
  // FORCE parameters
  int trainStep = 5;
  float trainRate = 4e5; //the smaller the faster the training occurs
  int trainStart = 0; //start training after trainStart*dt time
  int trainStop = T; //stop training here but keep net running
  
  // Tasks
  int numIter = 8; //number of training tasks
  vec samples = shuffle(linspace<vec>(1, numIter, numIter));
  mat stim, inp, tgt;
  samples.save("/home/neurociencia/discriminate/tasks.dat", raw_ascii);

  // How much time does it take to train our network?
  double timeTaken;

  fstream logfile("/home/neurociencia/discriminate/discriminate.log", fstream::out);
  
  logfile << "Iteration \t Time (s)" << endl;

  mat wSave, wOutSave;

  for (int i = 0; i < numIter; i++)
  {
    stim = stimulus((int)as_scalar(samples(i))%4, dt, T);
    inp = stim.rows(0, 1);
    tgt = stim.rows(2, 3);
    
    timer.tic();

    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, trainStart, trainStop, trainRate, saveRate, spikeTest, i);

    wSave = myNet.w;
    wSave.save("/home/neurociencia/discriminate/w_" + toString(i) + ".dat", raw_ascii);
    wOutSave = myNet.wOut;
    wOutSave.save("/home/neurociencia/discriminate/wOut_" + toString(i) + ".dat", raw_ascii);

    timeTaken = timer.toc();
    
    logfile << i+1 << " \t " << timeTaken << endl;
    cout << "Total time: " << timeTaken << endl;
  }

  logfile.close();

  return 0;
}

