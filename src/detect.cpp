#include "runSpikeNet.h"
using namespace arma;

int main(int argc, char* argv[]){

  wall_clock timer;

  /* True only for checking chaotic behaviour by removing one spike */
  bool spikeTest = false;
  bool rasterPlot = false;

  string basePath = "/home/neurociencia/detect/";
  string netPath = basePath + argv[1] + "/";
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
  float eqTime = 1.0;

  logfile << "Equilibrating... \n";
  //myNet = equilibrateSpikeNet(myNet, dt, eqTime, netPath, initPath, savePathEq, logfile);

  logfile << "Start training... \n";

  /* FORCE parameters */
  int trainStep = 5;
  int saveRate = 10;

  /* Load trials */
  mat trial, inp, tgt;
  int nTrialsTrain = 100;

  timelogs << "Time" << endl;

  /* Learning loop */

  for (int j = 0; j < nTrialsTrain; j++)
  {
    timer.tic();

    logfile << "\n";
    logfile << "Train " << j << " of " << nTrialsTrain << "\n";
    logfile << "\n";

    trial.load(trialsPath + "trial" + toString(j) + ".dat", raw_ascii);
    inp = trial.cols(0,2);
    tgt = trial.col(3);
    inp = inp.t();
    tgt = tgt.t();
   
    //myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, j, netPath, dynPath, savePathTrain, spikeTest, rasterPlot, logfile);
    
    timelogs << timer.toc() << endl;
  }

  timelogs.close();
  

  /* Test loop */
  trainStep = (int) INFINITY;
  int nTrialsTest = 100;

  for (int k = nTrialsTrain; k < nTrialsTrain+nTrialsTest; k++)
  {    
    timer.tic();

    logfile << "\n";
    logfile << "Train " << k << " of " << nTrialsTest << "\n";
    logfile << "\n";

    trial.load(trialsPath + "trial" + toString(k) + ".dat", raw_ascii);
    inp = trial.cols(0,2);
    tgt = trial.col(3);
    inp = inp.t();
    tgt = tgt.t();
   
    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, k, netPath, dynPath, savePathTest, spikeTest, rasterPlot, logfile);
    
    timelogs << timer.toc() << endl;
  }
  
  logfile.close();

  return 0;
}
