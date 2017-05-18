#include "runSpikeNet.h"
using namespace arma;

int main(){

  /* True only for checking chaotic behaviour by removing one spike */
  bool spikeTest = false;
  bool rasterPlot = false;

  string basePath = "/home/neurociencia/sine2Hz/";
  string netPath = basePath + toString("G_005_Q_30_l_4e5/");
  string savePath = netPath + toString("runMV/");

  int FORCEmode = 0;

  /* Create logfile */
  ofstream logfile((netPath + toString("logfile.log")).c_str());
  
  /* Create net from files */
  _Net myNet = loadSpikeNet(netPath);
  
  /* Equilibrium */
  float dt = 5e-4;
  float eqTime = .5;

  cout << "Equilibrating... \n";
  logfile << "Equilibrating... \n";

  myNet = equilibrateSpikeNet(myNet, dt, eqTime, FORCEmode, netPath, savePath, logfile);

  cout << "Start training... \n";
  logfile << "Start training... \n";

  /* FORCE parameters */
  int trainStep = 5;
  int saveRate = 1;

  /* Load trials */
  mat trial, inp, tgt;

  trial.load(basePath + "sine_dt_5e4.dat", raw_ascii);
  inp = trial.col(0);
  tgt = trial.col(1);
  inp = inp.t();
  tgt = tgt.t();

  /* Learning */
  myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, 1, FORCEmode, netPath, savePath, spikeTest, rasterPlot, logfile);

  cout << "Start testing... \n";
 /* Testing */
 myNet = runSpikeNet(myNet, inp, tgt, dt, (int)INFINITY, saveRate, 2, FORCEmode, netPath, savePath, spikeTest, rasterPlot, logfile);

  return 0;
}
