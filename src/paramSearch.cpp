#include <iostream>
#include <fstream>
#include <armadillo>
#include <string>
#include <sstream>
#include <cstdlib>
#include "createSpikeNet.h"
#include "runSpikeNet.h"
using namespace std;
using namespace arma;

int main()
{
  int N;
  float G, Q;

  float vth = -40.0;
  float vreset = -65.0;
  float vinf = -39.0;
  float tref = .002;
  float tm = .01;
  float td = .02;
  float tr = .002;

  float p = .1;
  int nIn = 1;
  int nOut = 1;
  
  int T = 100000;

  bool spikeTest = false;

  float dt = 5e-5;
  int trainStep = (int)INFINITY;
  float trainRate = 4e5;
  int trainStart = (int)T/5;
  int trainStop = (int)7*T/10;

  mat time = dt*linspace<vec>(0, T, T);
  mat inp = 0.0*time.t();
  mat tgt = sin(8.0*time).t();

  ifstream file("parameters.dat");
  
  if (file.is_open())
  {
    while (!file.eof())
    {
      file >> N >> G >> Q;

      vec v = (vth-vreset)*arma::randu<vec>(N)+vreset;
      vec r = arma::zeros<vec>(N);
      vec h = arma::randu<vec>(N);

      _Net myNet = createSpikeNet(vth, vreset, vinf, tref, tm, td, tr, N, p, nIn, nOut, G, Q, v, r, h);

      runSpikeNet(myNet, inp, tgt, dt, trainStep, trainStart, trainStop, trainRate, spikeTest);

    }
  }
  else cout << "Unable to open file" << endl;

  return 0;
}
