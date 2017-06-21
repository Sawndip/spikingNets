#include "runSpikeNet.h"
using namespace arma;

int main(int argc, char* argv[]){

  string basePath = "/home/neurociencia/svm/";
  string netPath = basePath + toString("sequential/") + argv[1] + "/";
  string savePathTrain = netPath + toString("train/");
  string savePathTest = netPath + toString("test/");
  string savePathEq = netPath + toString("eq/");
  string initPath = netPath + toString("init/");
  string dynPath = netPath + toString("dyn/");

  /* Load trials */
  int j = 19;
  mat trials, base, inp, tgt;
  
  trials.load(basePath + "y_geq_x_train.dat", raw_ascii);
  
  base.load(basePath + "base.dat", raw_ascii);
   
  inp = join_rows(join_rows(base.col(0), base.col(1)), trials(j, 0)*base.col(2) + trials(j, 1)*base.col(3));
  
  tgt = trials(j, 2)*base.col(4);
  
  inp.save(basePath + toString("checkInput.dat"), raw_ascii);
  tgt.save(basePath + toString("checkTarget.dat"), raw_ascii);

  return 0;
}
