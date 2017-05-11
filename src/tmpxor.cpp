#include "runSpikeNet.h"
using namespace arma;

int main(){

  /* True only for checking chaotic behaviour by removing one spike */
  bool spikeTest = false;
  bool rasterPlot = false;

  string basePath = "/home/neurociencia/tmpxor/";
  string netPath = basePath + toString("G_005_Q_30_l_4e5/");
  string savePathTrain = netPath + toString("train_100/");
  string savePathTest = netPath + toString("test_100/");

  /* Create logfile */
  ofstream logfile((netPath + toString("logfile.log")).c_str());
  
  /* Create net from files */
  _Net myNet = loadSpikeNet(netPath);
  
  /* Equilibrium */
  float dt = 5e-5;
  float eqTime = .5;

  logfile << "Equilibrating... \n";
  myNet = equilibrateSpikeNet(myNet, dt, eqTime, netPath, savePathTrain, logfile);

  logfile << "Start training... \n";
  /* Integration */
  int T = 40000;
  float totalTime = T*dt;

  /* FORCE parameters */
  int trainStep = 5;
  int saveRate = 100;


  /* Load trials */
  mat trial, inp, tgt;
  ivec trials;
  trials.load(basePath + "tmpxorTrain100.dat", raw_ascii);


  /* Learning loop */

  for (int j=0; j < trials.n_elem; j++)
  {
    logfile << "\n";
    logfile << "Train " << j << " of " << trials.n_elem << "\n";
    logfile << "\n";

 trial.load(basePath + "tmpxor" + toString(as_scalar(trials.row(j))) + ".dat", raw_ascii);

    inp = trial.col(0);
    tgt = trial.col(1);
    inp = inp.t();
    tgt = tgt.t();

    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, j, netPath, savePathTrain, spikeTest, rasterPlot, logfile);
  }

  
  /* Load tests */
  mat test;
  ivec tests;
  tests.load(basePath + "tmpxorTests100.dat", raw_ascii);


  /* Test loop */
  trainStep = (int) INFINITY;

  for (int k = 0; k < tests.n_elem; k++)
  {
    logfile << "\n";
    logfile << "Test " << k << " of " << tests.n_elem << "\n";
    logfile << "\n";

    test.load(basePath + "tmpxor" + toString(as_scalar(tests.row(k))) + ".dat", raw_ascii);

    inp = test.col(0);
    tgt = test.col(1);
    inp = inp.t();
    tgt = tgt.t();

    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, k, netPath, savePathTest, spikeTest, rasterPlot, logfile);
  }
  
  return 0;
}
