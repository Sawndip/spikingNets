#include "runSpikeNet.h"
using namespace std;
using namespace arma;

int main(){
  
  arma_rng::set_seed(42);

  /* True only for checking chaotic behaviour by removing one spike */
  bool spikeTest = false;

  string netPath = "/home/neurociencia/tmpxor/G_005_Q_20_l_400/init/";
  string savePathTrain = "/home/neurociencia/tmpxor/G_005_Q_20_l_400/train_28/";
  string savePathTest = "/home/neurociencia/tmpxor/G_005_Q_20_l_400/test_28/";
  string loadPath = "/home/neurociencia/tmpxor/";


  /* Create net from files */
  _Net myNet = loadSpikeNet(netPath);

  
  /* Equilibrium */
  float dt = 5e-5;
  float eqTime = .5;

  cout << "Equilibrating..." << endl;
  myNet = equilibrateSpikeNet(myNet, dt, eqTime);

  cout << "Start training..." << endl;
  /* Integration */
  int T = 40000;
  float totalTime = T*dt;

  /* FORCE parameters */
  int trainStep = 5;
  int trainStart = 0;
  int trainStop = T;
  int saveRate = 100;
  int saveFORCE = T-1;


  /* Load trials */
  mat trial, inp, tgt;
  ivec trials;
  trials.load(loadPath + "tmpxorTrain28.dat", raw_ascii);


  /* Learning loop */

  for (int j=0; j < trials.n_elem; j++)
  {
    trial.load(loadPath + "tmpxor" + toString(as_scalar(trials.row(j))) + ".dat", raw_ascii);

    inp = trial.col(0);
    tgt = trial.col(1);
    inp = inp.t();
    tgt = tgt.t();

    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, trainStart, trainStop, saveRate, j, saveFORCE, savePathTrain, spikeTest);
  }

  
  /* Load tests */
  mat test;
  ivec tests;
  tests.load(loadPath + "tmpxorTests.dat", raw_ascii);


  /* Test loop */
  trainStep = (int) INFINITY;

  for (int k = 0; k < tests.n_elem; k++)
  {
    test.load(loadPath + "tmpxor" + toString(as_scalar(tests.row(k))) + ".dat", raw_ascii);

    inp = test.col(0);
    tgt = test.col(1);
    inp = inp.t();
    tgt = tgt.t();

    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, trainStart, trainStop, saveRate, k, saveFORCE, savePathTest, spikeTest);
  }
  
  return 0;
}
