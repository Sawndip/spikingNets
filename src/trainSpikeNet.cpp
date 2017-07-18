#include "runSpikeNet.h"
using namespace arma;

int main(int argc, char* argv[]){

  mat trial, inp, tgt;

  wall_clock timer;

  string basePath = toString("/home/neurociencia/") + argv[1] + "/"; 
  string netPath = basePath + argv[2] + "/";
  string trialsPath = basePath + argv[3] + "/";
  string trainPath = netPath + toString("train/");
  string testPath = netPath + toString("test/");
  string eqPath = netPath + toString("eq/");
  string initPath = netPath + toString("init/");
  string dynPath = netPath + toString("dyn/");

  fstream timelogs(realpath((netPath + toString("timelogs.log")).c_str(), NULL), fstream::out);
  timelogs << "Time" << endl;
  
  cout << "Creating net...\n";  
  _Net myNet = loadSpikeNet(netPath, initPath);
   
  cout << "Equilibrating...\n";
  myNet = equilibrateSpikeNet(myNet, netPath, initPath, eqPath);

  cout << "Start training...\n";
  
  /* TODO: determine #trials dynamically doing ls | wc -l or something */
  int nTrialsTrain = 280;
  int nTrialsTest = 56;

  /* Learning and testing loop */

  for (int j = 0; j < nTrialsTrain + nTrialsTest; j++)
  {
    timer.tic();

    cout << "\n";
    cout << "Train " << j << " of " << nTrialsTrain << "\n";
    cout << "\n";

    trial.load(realpath((trialsPath + "trial" + toString(j) + ".dat").c_str(), NULL), raw_ascii);
    
    if (myNet.nIn == 1)
    {
      inp = trial.col(0);
    }
    else
    {
      inp = trial.cols(0, myNet.nIn-1);
    }

    if (myNet.nOut == 1)
    {
      tgt = trial.col(trial.n_cols-1);
    }
    else
    {
      tgt = trial.cols(myNet.nIn, trial.n_cols-1);
    }

    inp = inp.t();
    tgt = tgt.t();
   
    if (j < nTrialsTrain)
    {
      myNet = runSpikeNet(myNet, netPath, dynPath, trainPath, inp, tgt, j, 5);
    }
    else
    {
      myNet = runSpikeNet(myNet, netPath, dynPath, testPath, inp, tgt, j);
    }
    
    timelogs << timer.toc() << endl;
  }

  timelogs.close();
  
  return 0;
}
