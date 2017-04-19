#include <typeinfo>
#include "mconvert.h"
#include "createSpikeNet.h"
#include "runSpikeNet.h"
#include <armadillo>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <sstream>
using namespace std;
using namespace arma;


mat stimulus(int pattern, float dt, int T)
{
  //Auxiliary vars
  //int tWait0 = round(as_scalar(2.0*(arma::randu<vec>(1)) + 2.0)/dt);
  int tWait0 = round(1.0/dt);
  mat wait0 = arma::zeros<mat>(1, tWait0);
  mat inter = arma::zeros<mat>(1, round(1.0/dt));
  mat b = sin(arma::linspace<vec>(0, datum::pi, round(1.0/dt))).t();
  mat wait = arma::zeros<mat>(1, round(1.0/dt));
  mat dec = arma::zeros<mat>(1, T - (wait0.size() + 3*b.size() + wait.size()));
  mat tgtWait = join_rows(join_rows(join_rows(wait0, inter), join_rows(inter, inter)), wait);

  //Possible inputs
  mat inp0 = join_rows(join_rows(wait0, join_rows(join_rows(inter, inter), inter)), join_rows(wait, dec)); //no bumps
  mat inp1 = join_rows(join_rows(wait0, join_rows(join_rows(b, inter), inter)), join_rows(wait, dec)); //bump at earlier time
  mat inp2 = join_rows(join_rows(wait0, join_rows(join_rows(inter, inter), b)), join_rows(wait, dec)); //bump at later time
  mat inp3 = join_rows(join_rows(wait0, join_rows(join_rows(b, inter), b)), join_rows(wait, dec)); //two bumps

  //Possible decisions
  mat dec0 = arma::ones<mat>(1, T); //no reaction, channel 1
  mat dec1 = join_rows(arma::ones<mat>(1, tgtWait.size()), 1.5*arma::ones<mat>(1, T - tgtWait.size())); //reaction, channel 1
  mat dec2 = dec0 - 2.0*arma::ones<mat>(1, T); //no reaction, channel 2
  mat dec3 = dec1 - 2.0*arma::ones<mat>(1, T); //reaction, channel 2
  
  //cout << "Inputs: " << inp0.size() << "\t" << inp1.size() << "\t" << inp2.size() << "\t" << inp3.size() << endl;
  //cout << "Decisions: " << dec0.size() << "\t" << dec1.size() << "\t" << dec2.size() << "\t" << dec3.size() << endl;
   
  mat stim = arma::zeros<mat>(4, T);

  if (pattern == 0)
  {
    // Grouped-Grouped
    stim = join_cols(join_cols(inp3, inp0), join_cols(dec1, dec2));
  }
  else if (pattern == 1)
  {
    // Grouped-Extended
    stim = join_cols(join_cols(inp1, inp2), join_cols(dec0, dec3));
  }
  else if (pattern == 2)
  {
    // Extended-Grouped
    stim = join_cols(join_cols(inp2, inp1), join_cols(dec0, dec3));
  }
  else if (pattern == 3)
  {
    // Extended-Extended
    stim = join_cols(join_cols(inp0, inp3), join_cols(dec1, dec2));  
  }
  else
  {
    cout << "mat stimulus must be 1-4 int" << endl;
  }

  return stim;
}

