#include "runSpikeNet.h"
using namespace arma;

int main(int argc, char* argv[])
{
  wall_clock timer;

  _Net myNet;  
  
  bool rasterPlot = true;
  bool spikeTest;

  string basePath = "/home/neurociencia/paramSearch/";
  string netPath = basePath + argv[1] + "/";
  string savePath = netPath + toString("data/");
  string initPath = netPath + toString("init/");
  string dynPath = netPath + toString("dyn/");
  string savePathEq = netPath + toString("eq/");

  ofstream logfile((netPath + toString("logfile.log")).c_str());
  fstream timelogs((netPath + toString("timelogs.log")).c_str(), fstream::out);

  float dt = 5e-4;
  float time = 5.0;
  int trainStep = (int)INFINITY;
  int saveRate = 1;
  int T = round(time/dt);


  mat dummyInp, dummyTgt;
  dummyInp.zeros(1, T);
  dummyTgt.zeros(1, T);

  timelogs << "Time" << endl;
  
  timer.tic();
  myNet = loadSpikeNet(netPath, initPath);
  spikeTest = false;
  myNet = runSpikeNet(myNet, dummyInp, dummyTgt, dt, trainStep, saveRate, 0, netPath, initPath, savePath, spikeTest, rasterPlot, logfile);
  timelogs << timer.toc() << endl;
  
  timer.tic();
  myNet = loadSpikeNet(netPath, initPath);
  spikeTest = true;
  myNet = runSpikeNet(myNet, dummyInp, dummyTgt, dt, trainStep, saveRate, 1, netPath, initPath, savePath, spikeTest, rasterPlot, logfile);
  timelogs << timer.toc() << endl;

  return 0;
}
