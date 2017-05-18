#include "runSpikeNet.h"
using namespace arma;

int main(){

  wall_clock timer;

  /* True only for checking chaotic behaviour by removing one spike */
  bool spikeTest = false;
  bool rasterPlot = false;

  string basePath = "/home/neurociencia/svm/";
  string netPath = basePath + toString("G_004_Q_10_l_5e5/");
  string savePathTrain = netPath + toString("train_100/");
  string savePathTest = netPath + toString("test_100/");

  /* Create logfile */
  ofstream logfile((netPath + toString("logfile.log")).c_str());
  fstream timelogs((netPath + toString("timelogs.log")).c_str(), fstream::out);

  /* Create net from files */
  _Net myNet = loadSpikeNet(netPath);
  
  /* Equilibrium */
  float dt = 5e-4;
  float eqTime = 1;
  int FORCEmode = 0;

  logfile << "Equilibrating... \n";
  //myNet = equilibrateSpikeNet(myNet, dt, eqTime, FORCEmode, netPath, savePathTrain, logfile);

  logfile << "Start training... \n";

  /* FORCE parameters */
  int trainStep = 5;
  int saveRate = 10;

  /* Load trials */
  mat trials, base, inp, tgt;
  trials.load(basePath + "y_geq_x_train.dat", raw_ascii);
  base.load(basePath + "base.dat", raw_ascii);
  
  double timeTaken;

  timelogs << "Iteration \t Time (s)" << endl;

  /* Learning loop */

  for (int j=0; j < trials.n_rows; j++)
  {
    timer.tic();

    logfile << "\n";
    logfile << "Train " << j << " of " << trials.n_elem << "\n";
    logfile << "\n";

    inp = join_rows(base.col(0), join_rows(trials(j, 0)*base.col(1), trials(j, 1)*base.col(1)));
    tgt = trials(j, 2)*base.col(2);
    inp = inp.t();
    tgt = tgt.t();

    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, j, FORCEmode, netPath, savePathTrain, spikeTest, rasterPlot, logfile);

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

    inp = join_rows(base.col(0), join_rows(trials(k, 0)*base.col(1), trials(k, 1)*base.col(1)));
    tgt = trials(k, 2)*base.col(2);
    inp = inp.t();
    tgt = tgt.t();

    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, k, FORCEmode, netPath, savePathTest, spikeTest, rasterPlot, logfile);
  }
  
  logfile.close();

  return 0;
}
