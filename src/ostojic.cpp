#include "runSpikeNet.h"
using namespace arma;

int main(int argc, char* argv[])
{
  wall_clock timer;

  bool rasterPlot = true;
  bool spikeTest = false;

  string basePath = "/home/neurociencia/paramSearch/";
  string netPath = basePath + argv[1] + "/";
  string savePath = netPath + toString("data2/");
  string initPath = netPath + toString("init/");
  string dynPath = netPath + toString("dyn/");
  string savePathEq = netPath + toString("eq/");

  ofstream logfile((netPath + toString("logfile.log")).c_str());
  fstream timelogs((netPath + toString("timelogs.log")).c_str(), fstream::out);

  float dt = 5e-4;
  float eqTime = 1.5;
  float time = 5.0;
  int trainStep = (int)INFINITY;
  int saveRate = 1;
  int j = 1;
  int T = round(time/dt);

  timelogs << "Time" << endl;

  mat dummyInp, dummyTgt;
  dummyInp.zeros(1, T);
  dummyTgt.zeros(1, T);

  ifstream params((basePath + toString("G2.dat")).c_str());
  
  if (params.is_open())
  {
    while (!params.eof())
    {
      timer.tic();
  
      _Net myNet = loadSpikeNet(netPath, initPath);
      
      params >> myNet.G;
      std::cout << "G: " << myNet.G << endl;

      myNet = equilibrateSpikeNet(myNet, dt, eqTime, netPath, initPath, savePathEq, logfile);

      myNet = runSpikeNet(myNet, dummyInp, dummyTgt, dt, trainStep, saveRate, j, netPath, dynPath, savePath, spikeTest, rasterPlot, logfile);
      
      std::cout << "Time: " << timer.toc() << endl;
      timelogs << timer.toc() << endl;

      j = j+1;
    }
  }
  else cout << "Unable to open file" << endl;

  return 0;
}
