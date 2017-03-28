#define INFINITY
#include <typeinfo>
#include "mconvert.h"
#include "createSpikeNet.h"
#include "runSpikeNet.h"
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
  float vth = -40.0;
  float vreset = -65.0;
  float vinf = -39.0;
  float tref = .002; //refractory time
  float tm = .01; //membrane relaxation
  float td = .03; //decrease
  float tr = .002; //rise

  int N = 2000; //neurons
  float p = .1; //sparsity
  int nIn = 1; //inputs
  int nOut = 1; //outputs
  float G = 1.0;
  float Q = 20.0;

  int T = 10000; //total time in units of dt

  vec v = (vth-vreset)*arma::randu<vec>(N)+vreset;
  vec r = arma::zeros<vec>(N);
  vec h = arma::randu<vec>(N);
  
  bool spikeTest = false;

  _Net myNet = createSpikeNet(vth, vreset, vinf, tref, tm, td, tr, N, p, nIn, nOut, G, Q, v, r, h);
  
  float dt = 5e-5;
  int trainStep = (int)INFINITY;
  float trainRate = 4e5; //the smaller the faster the training occurs
  int trainStart = (int)T/5; //start training after trainStart*dt time
  int trainStop = (int)7*T/10; //stop training here but keep net running

  mat time = dt*linspace<vec>(0, T, T);
  mat inp = uhlenbeck(1.0, 0.0, 1.0, dt, T);
  //mat tgt = sin(8.0*datum::pi*time).t() + 2.0*cos(4.0*datum::pi*time).t();
  //tgt.save("tgt.dat", raw_ascii);
  inp.save("uhl.dat", raw_ascii);

  //runSpikeNet(myNet, inp, tgt, dt, trainStep, trainStart, trainStop, trainRate, spikeTest);
  
  return 0;
}
