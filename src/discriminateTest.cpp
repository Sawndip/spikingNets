#define INFINITY
#include <typeinfo>
#include "mconvert.h"
#include "createSpikeNet.h"
#include "runSpikeNet.h"
#include <armadillo>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <sstream>
using namespace std;
using namespace arma;


mat stimulus(int pattern, float dt, int T)
{
  //Auxiliary vars
  int tWait0 = round(as_scalar(2.0*(arma::randu<vec>(1)) + 2.0)/dt);
  mat wait0 = arma::zeros<mat>(1, tWait0);
  mat inter = arma::zeros<mat>(1, round(1.0/dt));
  mat b = sin(arma::linspace<vec>(0, datum::pi, round(1.0/dt))).t();
  mat wait = arma::zeros<mat>(1, round(2.0/dt));
  mat dec = arma::zeros<mat>(1, T - (wait0.size() + 3*b.size() + wait.size()));
  mat tgtWait = join_rows(join_rows(join_rows(wait0, inter), join_rows(inter, inter)), wait);

  //Possible inputs
  mat inp0 = join_rows(join_rows(wait0, join_rows(join_rows(inter, inter), inter)), join_rows(wait, dec)); //no bumps
  mat inp1 = join_rows(join_rows(wait0, join_rows(join_rows(b, inter), inter)), join_rows(wait, dec)); //bump at earlier time
  mat inp2 = join_rows(join_rows(wait0, join_rows(join_rows(inter, inter), b)), join_rows(wait, dec)); //bump at later time
  mat inp3 = join_rows(join_rows(wait0, join_rows(join_rows(b, inter), b)), join_rows(wait, dec)); //two bumps

  //Possible decisions
  mat dec0 = arma::ones<mat>(1, T); //no reaction, channel 1
  mat dec1 = join_rows(arma::ones<mat>(1, tgtWait.size()), 1.5*arma::ones<mat>(1, T - tgtWait.size())); //reaction, channel 1
  mat dec2 = dec0 - 2.0*arma::ones<mat>(1, T); //no reaction, channel 2
  mat dec3 = dec1 - 2.0*arma::ones<mat>(1, T); //reaction, channel 2
  
  cout << "Inputs: " << inp0.size() << "\t" << inp1.size() << "\t" << inp2.size() << "\t" << inp3.size() << endl;
  cout << "Decisions: " << dec0.size() << "\t" << dec1.size() << "\t" << dec2.size() << "\t" << dec3.size() << endl;
   
  mat stim = arma::zeros<mat>(4, T);

  if (pattern == 0)
  {
    // Grouped-Grouped
    stim = join_cols(join_cols(inp3, inp0), join_cols(dec1, dec2));
  }
  else if (pattern == 1)
  {
    // Grouped-Extended
    stim = join_cols(join_cols(inp1, inp2), join_cols(dec0, dec3));
  }
  else if (pattern == 2)
  {
    // Extended-Grouped
    stim = join_cols(join_cols(inp2, inp1), join_cols(dec0, dec3));
  }
  else if (pattern == 3)
  {
    // Extended-Extended
    stim = join_cols(join_cols(inp0, inp3), join_cols(dec1, dec2));  
  }
  else
  {
    cout << "mat stimulus must be 1-4 int" << endl;
  }

  return stim;
}

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


int main()
{
  arma_rng::set_seed(24);
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
  float G = .06;
  float Q = 5.0;

  // Initial values
  mat w;
  w.load("/home/javier/spikingNets/shell/w_0123.dat");
  mat wIn = 2.0*arma::randu<mat>(N, nIn) - 1.0*arma::ones<mat>(N, nIn); //uniform [-1,1]
  mat wOut;
  wOut.load("/home/javier/spikingNets/shell/wOut_0123.dat");
  mat wFb = 2.0*arma::randu<mat>(N, nOut) - 1.0*arma::ones<mat>(N, nOut); //uniform [-1,1]
  vec v = (vth-vreset)*arma::randu<vec>(N)+vreset;
  vec r = arma::zeros<vec>(N);
  vec h = arma::randu<vec>(N);
  
  // Initialize net
  _Net myNet = createSpikeNet(vth, vreset, vinf, tref, tm, td, tr, N, p, nIn, nOut, G, Q, w, wIn, wOut, wFb, v, r, h);
 
  //Integration parameters
  float totalTime = 10; //total time in seconds
  float dt = 5e-4;
  int T = (int)totalTime/dt; //total time in units of dt
  int saveRate = 20;

  // True only for checking chaotic behaviour by removing one spike
  bool spikeTest = false;
 
  // FORCE parameters
  int trainStep = (int)INFINITY;
  float trainRate = 4e5; //the smaller the faster the training occurs
  int trainStart = 0; //start training after trainStart*dt time
  int trainStop = T; //stop training here but keep net running
  
  // Tasks
  int numIter = 1; //number of training tasks
  vec samples = shuffle(linspace<vec>(1, numIter, numIter));
  mat stim, inp, tgt;
  samples.save("/home/neurociencia/discriminate/tasks_test.dat", raw_ascii);

  // How much time does it take to train our network?
  double timeTaken;

  ofstream logfile, paramsfile;
  
  //paramsfile.open("home/neurociencia/discriminate/parameters.dat");
  //paramsfile << "N \t G \t Q \t lambda \t dt" << endl;
  //paramsfile << N << "\t" << G "\t" << Q << "\t" << trainRate << "\t" << dt << endl;
  //paramsfile.close();


  ostringstream fileID;

  for (int i = 0; i < numIter; i++)
  {
    fileID << i;

    stim = stimulus((int)as_scalar(samples(i))%4, dt, T);
    inp = stim.rows(0, 1);
    tgt = stim.rows(2, 3);
    
    timer.tic();

    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, trainStart, trainStop, trainRate, saveRate, spikeTest, i);
  
    mat wSave = myNet.w;
    mat wOutSave = myNet.wOut;

    wSave.save("w_" + fileID.str() + "_test.dat", raw_ascii);
    wOutSave.save("wOut_" + fileID.str() + "_test.dat", raw_ascii);

    timeTaken = timer.toc();
    
    logfile << i << "\t" << timeTaken << endl;
  }

  logfile.close();

  return 0;
}
