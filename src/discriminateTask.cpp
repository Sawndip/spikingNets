#define INFINITY
#include <typeinfo>
#include "mconvert.h"
#include "createSpikeNet.h"
#include "runSpikeNet.h"
#include <armadillo>
using namespace std;
using namespace arma;


mat stimulus(int pattern, float dt, int T)
{
  int tWait0 = round(as_scalar(2.0*(arma::randu<vec>(1)) + 2.0)/dt);
  mat wait0 = arma::zeros<mat>(1, tWait0);
  mat inter = arma::zeros<mat>(1, round(1.0/dt));
  mat b = sin(arma::linspace<vec>(0, datum::pi, round(1.0/dt))).t();
  mat wait = arma::zeros<mat>(1, round(2.0/dt));
  mat tgtWait = join_rows(join_rows(join_rows(wait0, inter), join_rows(inter, inter)), wait);
  mat dec = arma::zeros<mat>(1, T - tgtWait.size());

  mat inp0 = join_rows(join_rows(inter, inter), inter);
  mat inp1 = join_rows(join_rows(b, inter), inter);
  mat inp2 = join_rows(join_rows(inter, inter), b);
  mat inp3 = join_rows(join_rows(b, inter), b);

  mat stim = arma::zeros<mat>(4, T);

  if (pattern == 1)
  {
    stim.row(1).cols(tWait0, tWait0 + b.size() - 1) = b;
  }
  else if (pattern == 2)
  {  
    stim = join_cols(join_cols(join_rows(join_rows(wait0, inp1), join_rows(wait, dec)),
                                       join_rows(join_rows(wait0, inp2), join_rows(wait, dec))),
                             join_cols(join_rows(tgtWait, dec) + 1.0,
                                       join_rows(tgtWait, dec + .5) - 1.0));
  }
  else if (pattern == 3)
  {  
    stim = join_cols(join_cols(join_rows(join_rows(wait0, inp2), join_rows(wait, dec)),
                                       join_rows(join_rows(wait0, inp1), join_rows(wait, dec))),
                             join_cols(join_rows(tgtWait, dec) + 1.0,
                                       join_rows(tgtWait, dec + .5) - 1.0));
  }
  else if (pattern == 4)
  {  
    stim = join_cols(join_cols(join_rows(join_rows(wait0, inp0), join_rows(wait, dec)),
                                       join_rows(join_rows(wait0, inp3), join_rows(wait, dec))),
                             join_cols(join_rows(tgtWait, dec + .5) + 1.0,
                                       join_rows(tgtWait, dec) - 1.0));
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
  wall_clock timer;

  float vth = -40.0;
  float vreset = -65.0;
  float vinf = -39.0;
  float tref = .002; //refractory time
  float tm = .01; //membrane relaxation
  float td = .03; //decrease
  float tr = .002; //rise

  int N = 2000; //neurons
  float p = .1; //sparsity
  int nIn = 2; //inputs
  int nOut = 2; //outputs
  float G = 1.0;
  float Q = 20.0;

  int T = 200000; //total time in units of dt

  vec v = (vth-vreset)*arma::randu<vec>(N)+vreset;
  vec r = arma::zeros<vec>(N);
  vec h = arma::randu<vec>(N);
  
  bool spikeTest = false;

  _Net myNet = createSpikeNet(vth, vreset, vinf, tref, tm, td, tr, N, p, nIn, nOut, G, Q, v, r, h);
  
  float dt = 5e-5;
  int trainStep = 50;
  float trainRate = 4e2; //the smaller the faster the training occurs
  int trainStart = 0; //start training after trainStart*dt time
  int trainStop = T; //stop training here but keep net running

  int pattern = 1;
  mat stim = stimulus(pattern, dt, T);
  mat inp = stim.rows(0, 1);
  mat tgt = stim.rows(2, 3);

  timer.tic();

  runSpikeNet(myNet, inp, tgt, dt, trainStep, trainStart, trainStop, trainRate, spikeTest);
  
  double timeTaken = timer.toc();
  cout << "Total time (2e5 iterations with FORCE all through): " << timeTaken << " seconds" << endl;

  return 0;
}
