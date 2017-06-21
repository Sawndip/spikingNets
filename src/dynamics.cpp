#include "runSpikeNet.h"
using namespace arma;

int main(){

  wall_clock timer;

  /* True only for checking chaotic behaviour by removing one spike */
  bool spikeTest = false;
  bool rasterPlot = false;

  string basePath = "/home/neurociencia/dynamics/";
  string netPath = basePath + toString("net7/");
  string savePathEq = netPath + toString("eq/");
  string savePathTrain = netPath + toString("train/");
  string savePathTest = netPath + toString("test/");
  string initPath = netPath + toString("init/");
  string dynPath = netPath + toString("dyn/");

  /* Create logfile */
  ofstream logfile((netPath + toString("logfile.log")).c_str());
  fstream timelogs((netPath + toString("timelogs.log")).c_str(), fstream::out);

  /* Create net from files */
  _Net myNet = loadSpikeNet(netPath, initPath);

  /* Equilibrium */
  float dt = 5e-4;
  float eqTime = 1;

  logfile << "Equilibrating... \n";
  myNet = equilibrateSpikeNet(myNet, dt, eqTime, netPath, initPath, savePathEq, logfile);
  
  /* FORCE */
  int saveRate = 1;
  int trainStep = 5;
  mat train, test, inp, tgt;
  
  double timeTaken;
  timelogs << "Iteration \t Time" << endl;


  /****** DYNAMICS 1 - 5Hz sine ******/
  /* Training */
  logfile << "Start training... \n";
  
  train.load(basePath + toString("dyn1.dat"), raw_ascii);
  inp = train.col(0);
  tgt = train.col(1);
  inp = inp.t();
  tgt = tgt.t();

  timer.tic();

  myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, 1, netPath, initPath, savePathTrain, spikeTest, rasterPlot, logfile);

  timeTaken = timer.toc();
  timelogs << timeTaken << endl;

  /* Testing */
  logfile << "Start testing... \n";

  timer.tic();

  myNet = runSpikeNet(myNet, inp, tgt, dt, (int)INFINITY, saveRate, 1, netPath, dynPath, savePathTest, spikeTest, rasterPlot, logfile);

  timeTaken = timer.toc();
  timelogs << timeTaken << endl;


  /******    DYNAMICS 2 - Sawtooth  *******/
  logfile << "Start training... \n";
  
  train.load(basePath + toString("dyn2.dat"), raw_ascii);
  inp = train.col(0);
  tgt = train.col(1);
  inp = inp.t();
  tgt = tgt.t();

  timer.tic();

  myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, 2, netPath, initPath, savePathTrain, spikeTest, rasterPlot, logfile);

  timeTaken = timer.toc();
  timelogs << timeTaken << endl;

  /* Testing */
  logfile << "Start testing... \n";

  timer.tic();

  myNet = runSpikeNet(myNet, inp, tgt, dt, (int)INFINITY, saveRate, 2, netPath, dynPath, savePathTest, spikeTest, rasterPlot, logfile);

  timeTaken = timer.toc();
  timelogs << timeTaken << endl;


  /******    DYNAMICS 3 - Sum of sines  *******/
  logfile << "Start training... \n";
  
  train.load(basePath + toString("dyn3.dat"), raw_ascii);
  inp = train.col(0);
  tgt = train.col(1);
  inp = inp.t();
  tgt = tgt.t();

  timer.tic();

  myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, 3, netPath, initPath, savePathTrain, spikeTest, rasterPlot, logfile);

  timeTaken = timer.toc();
  timelogs << timeTaken << endl;

  /* Testing */
  logfile << "Start testing... \n";

  timer.tic();

  myNet = runSpikeNet(myNet, inp, tgt, dt, (int)INFINITY, saveRate, 3, netPath, dynPath, savePathTest, spikeTest, rasterPlot, logfile);

  timeTaken = timer.toc();
  timelogs << timeTaken << endl;


  /******    DYNAMICS 4 - Noisy sine  *******/
  logfile << "Start training... \n";
  
  train.load(basePath + toString("dyn4.dat"), raw_ascii);
  inp = train.col(0);
  tgt = train.col(1);
  inp = inp.t();
  tgt = tgt.t();

  timer.tic();

  myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, 4, netPath, initPath, savePathTrain, spikeTest, rasterPlot, logfile);

  timeTaken = timer.toc();
  timelogs << timeTaken << endl;

  /* Testing */
  logfile << "Start testing... \n";

  timer.tic();

  myNet = runSpikeNet(myNet, inp, tgt, dt, (int)INFINITY, saveRate, 4, netPath, dynPath, savePathTest, spikeTest, rasterPlot, logfile);

  timeTaken = timer.toc();
  timelogs << timeTaken << endl;


  timelogs.close();
  logfile.close();

  return(0);
}

