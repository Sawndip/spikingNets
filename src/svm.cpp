#include "runSpikeNet.h"
using namespace arma;

int main(int argc, char* argv[]){

  wall_clock timer;

  /* True only for checking chaotic behaviour by removing one spike */
  bool spikeTest = false;
  bool rasterPlot = false;

  string basePath = "/home/neurociencia/svm/";
  string netPath = basePath + argv[1] + "/" + argv[2] + "/";
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
  myNet = equilibrateSpikeNet(myNet, dt, eqTime, netPath, initPath, savePathEq, logfile);

  logfile << "Start training... \n";

  /* FORCE parameters */
  int trainStep = 5;
  int saveRate = 10;

  /* Load trials */
  mat trials, base, inp, tgt;
  trials.load(basePath + "y_geq_x_train.dat", raw_ascii);
  base.load(basePath + "base_" + argv[1] + ".dat", raw_ascii);
  double timeTaken;

  timelogs << "Iteration \t Time (s)" << endl;

  /* Learning loop */

  for (int j = 0; j < trials.n_rows; j++)
  {
    timer.tic();

    logfile << "\n";
    logfile << "Train " << j << " of " << trials.n_elem << "\n";
    logfile << "\n";

    if (string(argv[1]) == "simultaneous")
    {
      inp = join_rows(join_rows(base.col(0), base.col(1)), join_rows(trials(j, 0)*base.col(2), trials(j, 1)*base.col(3)));
    }
    else if (string(argv[1]) == "sequential")
    {
      inp = join_rows(join_rows(base.col(0), base.col(1)), trials(j, 0)*base.col(2)+trials(j, 1)*base.col(3));
    }
    
    tgt = trials(j, 2)*base.col(4);
    
    inp = inp.t();
    tgt = tgt.t();
    
    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, j, netPath, dynPath, savePathTrain, spikeTest, rasterPlot, logfile);
    
    timeTaken = timer.toc();
    timelogs << j+1 << " \t " << timeTaken << endl;
  }

  timelogs.close();
  
  /* Load tests */
  mat tests;
  tests.load(basePath + "y_geq_x_test.dat", raw_ascii);

  /* Test loop */
  trainStep = (int) INFINITY;

  for (int k = 0; k < tests.n_elem; k++)
  {
    logfile << "\n";
    logfile << "Test " << k << " of " << tests.n_elem << "\n";
    logfile << "\n";
    
    if (string(argv[1]) == "simultaneous")
    {
      inp = join_rows(join_rows(base.col(0), base.col(1)), join_rows(trials(k, 0)*base.col(2), trials(k, 1)*base.col(3)));
    }
    else if (string(argv[1]) == "sequential")
    {
      inp = join_rows(join_rows(base.col(0), base.col(1)), trials(k, 0)*base.col(2)+trials(k, 1)*base.col(3));
    }
    
    tgt = trials(k, 2)*base.col(4);
    
    inp = inp.t();
    tgt = tgt.t();

    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, k, netPath, dynPath, savePathTest, spikeTest, rasterPlot, logfile);
  }
  
  logfile.close();

  return 0;
}
