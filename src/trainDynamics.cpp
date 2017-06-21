#include "runSpikeNet.h"
using namespace arma;

int main(int argc, char* argv[]){

  wall_clock timer;

  /* True only for checking chaotic behaviour by removing one spike */
  bool spikeTest = false;
  bool rasterPlot = false;

  string basePath = toString("/home/neurociencia/") + argv[1] + "/";
  
  string netPath = basePath + argv[2] + "/";
  string trialsPath = basePath + "trials/";
  string savePathTrain = netPath + toString("train/");
  string savePathTest = netPath + toString("test/");
  string savePathEq = netPath + toString("eq/");
  string initPath = netPath + toString("init/");
  string dynPath = netPath + toString("dyn/");

  /* Create logfile */
  ofstream logfile((netPath + toString("logfile.log")).c_str());
  fstream timelogs((netPath + toString("timelogs.log")).c_str(), fstream::out);

  /* Create net from files */
  _Net myNet = loadSpikeNet(netPath, initPath);
  
  /* Equilibrium */
  float dt = 5e-4;
  float eqTime = 1.5;

  logfile << "Equilibrating... \n";
  myNet = equilibrateSpikeNet(myNet, dt, eqTime, netPath, initPath, savePathEq, logfile);

  logfile << "Start training... \n";

  /* TODO: Parametrise FORCE parameters - move them to input file */
  int trainStep;
  int saveRate = 10;

  /* Load trials */
  mat trial, inp, tgt;
  
  /* TODO: determine #trials dynamically doing ls | wc -l or something */
  int nTrialsTrain = 100;

  timelogs << "Time" << endl;

  trial.load(trialsPath + "trial0.dat", raw_ascii);
  inp = trial.col(0);
  tgt = trial.col(1);
  inp = inp.t();
  tgt = tgt.t();

  /* TRAIN */
  timer.tic();

  trainStep = 5;
  myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, 0, netPath, dynPath, savePathTrain, spikeTest, rasterPlot, logfile);
 
  timelogs << timer.toc() << endl;
  timer.tic();

  /* TEST */
  trainStep = (int)INFINITY;
  myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, 0, netPath, dynPath, savePathTest, spikeTest, rasterPlot, logfile);
  
  timelogs << timer.toc() << endl;
  
  logfile.close();
  timelogs.close();

  return 0;
}
