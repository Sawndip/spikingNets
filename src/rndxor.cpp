#include "runSpikeNet.h"
using namespace arma;

int main(int argc, char* argv[]){

  wall_clock timer;

  /* True only for checking chaotic behaviour by removing one spike */
  bool spikeTest = false;
  bool rasterPlot = false;

  string basePath = "/home/neurociencia/rndxor/";
  string netPath = basePath + toString("inputAndDecision/") + argv[1] + "/";
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
  int saveRate = 10;
  int trainStep = 5;
  
  /* I/O */
  mat trials, delays, thisTrial, auxInp, auxTgt, inp, tgt, tmp1, tmp2, tmp3, tmp4;
  int delInp, delTgt;
 

  /* Learning loop */
  logfile << "Start training... \n";

  timelogs << "sec" << endl;

  trials.load(basePath + toString("train_trials.dat"), raw_ascii);
  delays.load(basePath + toString("train_delays.dat"), raw_ascii);
  
  for (int j=0; j < trials.n_elem; j++)
  {
    timer.tic();

    logfile << "\n";
    logfile << "Train " << j << " of " << trials.n_elem << "\n";

    auxInp.load(basePath + "inp" + toString(as_scalar(trials.row(j))) + ".dat", raw_ascii);
    auxTgt.load(basePath + "tgt" + toString(as_scalar(trials.row(j))) + ".dat", raw_ascii);

    auxInp = auxInp.t();
    auxTgt = auxTgt.t();

    delInp = as_scalar(delays(j,0));
    delTgt = as_scalar(delays(j,1));
    
    tmp1 = zeros(2, delInp);
    tmp2 = zeros(2, delTgt);
    tmp3 = zeros(1, delInp + auxInp.n_cols + delTgt);
    tmp4 = zeros(3, auxTgt.n_cols);
        
    inp = join_rows(join_cols(join_rows(join_rows(tmp1, auxInp), tmp2), tmp3), tmp4);
    tgt = join_rows(tmp3, auxTgt.row(1));

    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, j, netPath, dynPath, savePathTrain, spikeTest, rasterPlot, logfile);

    timelogs << timer.toc() << endl;
  }

 
  /* Testing loop */

  logfile << "Start training... \n";

  trials.load(basePath + toString("test_trials.dat"), raw_ascii);
  delays.load(basePath + toString("test_delays.dat"), raw_ascii);
  
  for (int k=0; k < trials.n_elem; k++)
  {
    timer.tic();

    logfile << "\n";
    logfile << "Train " << k << " of " << trials.n_elem << "\n";

    auxInp.load(basePath + "inp" + toString(as_scalar(trials.row(k))) + ".dat", raw_ascii);
    auxTgt.load(basePath + "tgt" + toString(as_scalar(trials.row(k))) + ".dat", raw_ascii);

    auxInp = auxInp.t();
    auxTgt = auxTgt.t();

    delInp = as_scalar(delays(k,0));
    delTgt = as_scalar(delays(k,1));

    tmp1 = zeros(2, delInp);
    tmp2 = zeros(2, delTgt);
    tmp3 = zeros(1, delInp + auxInp.n_cols + delTgt);
    tmp4 = zeros(3, auxTgt.n_cols);
        
    inp = join_rows(join_cols(join_rows(join_rows(tmp1, auxInp), tmp2), tmp3), tmp4);
    tgt = join_rows(tmp3, auxTgt.row(1));

    myNet = runSpikeNet(myNet, inp, tgt, dt, trainStep, saveRate, k, netPath, dynPath, savePathTest, spikeTest, rasterPlot, logfile);

    timelogs << timer.toc() << endl;
  }

  timelogs.close();
  logfile.close();

  return(0);
}
