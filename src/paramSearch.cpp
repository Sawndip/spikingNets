#include "runSpikeNet.h"
using namespace arma;

int main()
{
  wall_clock timer;

  bool rasterPlot = false;
  bool spikeTest = false;

  string basePath = "/home/neurociencia/paramSearch/";
  string netPath = basePath + toString("net1/");
  string savePathTrain = netPath + toString("train/");
  string savePathTest = netPath + toString("test/");
  string savePathEq = netPath + toString("eq/");
  string initPath = netPath + toString("init/");
  string dynPath = netPath + toString("dyn/");

  ofstream logfile((netPath + toString("logfile.log")).c_str());
  fstream timelogs((netPath + toString("timelogs.log")).c_str(), fstream::out);

  float dt = 5e-4;
  float eqTime = 1.0;
  int trainStep = 5;
  int saveRate = 10;
  int j = 1;

  mat train, test, inp, tgt;
  train.load(basePath + "train_data.dat");
  test.load(basePath + "test_data.dat");

  double timeTaken;
  timelogs << "Time" << endl;

  ifstream params((basePath + toString("parameters.dat")).c_str());
  
  if (params.is_open())
  {
    while (!params.eof())
    {
      timer.tic();
  
      _Net myNet = loadSpikeNet(netPath, initPath);
      
      params >> myNet.G >> myNet.Q >> myNet.lambda;

      /* Equilibrium */
      myNet = equilibrateSpikeNet(myNet, dt, eqTime, netPath, initPath, savePathEq, logfile);

      /* Training */
      inp = train.cols(0,2);
      tgt = train.col(3);
      inp = inp.t();
      tgt = tgt.t();

      myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, j, netPath, dynPath, savePathTrain, spikeTest, rasterPlot, logfile);
      
      /* Testing */
      inp = test.cols(0,2);
      tgt = test.col(3);
      inp = inp.t();
      tgt = tgt.t();
      
      myNet = runSpikeNet(myNet, inp, tgt, dt, (int)INFINITY, saveRate, j, netPath, dynPath, savePathTest, spikeTest, rasterPlot, logfile);

      timeTaken = timer.toc();
      timelogs << timeTaken << endl;

      j = j+1;
    }
  }
  else cout << "Unable to open file" << endl;

  return 0;
}
