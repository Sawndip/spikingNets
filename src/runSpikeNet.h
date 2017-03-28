// Automatically translated using m2cpp 0.5 on 2017-02-27 19:06:26
//
// This function takes an already created network (_Net object) and integrates
// its dynamics with timestep dt given a certain input.

#ifndef RUNSPIKENET_M_HPP
#define RUNSPIKENET_M_HPP

#include "mconvert.h"
#include "createSpikeNet.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <armadillo>
using namespace std;
using namespace arma;

void runSpikeNet(_Net net, mat input, mat tgt, float dt, int trainStep, int trainStart, int trainStop, float trainRate, bool spikeTest)
{ 
  // note that trainStep should be given in units of dt, e.g. if integration
  // time is dt = 0.001 seconds and the training occurs every 0.05 seconds,
  // then int trainStep = 50;
  // Setting int trainStep = INFINITY the network runs without training, so
  // this function can be used to initialize it too.
  
  //LIF neurons' parameters
  float vth = net.vth;
  float vreset = net.vreset;
  float vinf = net.vinf;
  float tref = net.tref;
  float tm = net.tm;
  float td = net.td;
  float tr = net.tr;

  //Network parameters
  int N = net.N;
  float G = net.G;
  float Q = net.Q;

  //Network initial values
  vec v = net.v;
  vec r = net.r;
  vec h = net.h;
  mat wIn = net.wIn;
  mat wOut = net.wOut;
  mat wFb = net.wFb;
  mat w = G*net.w + Q*wFb*wOut;
  int T = input.n_cols; //time duration of input
  
  //FORCE initial values
  mat P; P.eye(N, N)/trainRate;
  mat err = arma::zeros<mat>(wOut.n_rows, N);
  
  //Save time evolution of variables here
  mat vSave = v;
  mat rSave = r;
  mat hSave = h;
  mat wOutSave = wOut;
  mat spikes = arma::zeros<mat>(T, N); //spikes[i,j] = 1 if neuron i spiked at time j - 0 otherwise

  //Declare increments for integration
  vec ref = arma::zeros<vec>(N);
  vec dv = arma::zeros<vec>(N);
  vec dr = arma::zeros<vec>(N);
  vec dh = arma::zeros<vec>(N);

  //Test for chaos deleting spike
  bool spikeDeleted = false;
  int deleteTime = (int)(T/2.0); //delete a spike only in the second half of the interval

  for (int i=0; i<T; i=i+1)
  {
    //Keep track of variable evolution every 100 steps
    if (i%100 == 0 && i > 0)
    {
      //vSave = join_rows(vSave, v);
      rSave = join_rows(rSave, r);
      //hSave = join_rows(hSave, h);
      wOutSave = join_cols(wOutSave, wOut);
    }
    
    if (i%500 == 0)
    {
      cout << "Iteration " << i << " out of " << T << "\t" << (float)(100*i/T) << " o/o progress" << endl;
    }
    
    //LIF EDOs system
    dv = (-v + vinf + w*r + wIn*input.col(i))/tm; //voltage
    v.elem(find(ref <= 0)) = v.elem(find(ref <= 0)) + dv.elem(find(ref <= 0))*dt; //update neurons not in refractory period

    //Double exponential filter
    dr = -r/td + h;
    r = r + dt*dr;
    
    dh = -h/tr; //h var
    h = h + dh*dt + conv_to<vec>::from(v > vth)/(tr*td); //update h
    
    if (spikeTest && !spikeDeleted && (i == deleteTime))
    {
      uvec deletedNeuron = find(v > vth, 1, "first"); //
      v.elem(deletedNeuron).fill(vreset);
      spikeDeleted = true;
    }

    //Fire prescription
    //spikes.row(i) = (conv_to<vec>::from(v > vth)).t(); //keep track of spikes before updating
    ref.elem(find(ref > 0)) = ref.elem(find(ref > 0)) - dt; //subtract the elapsed timestep from neurons in refractory period
    ref.elem(find(v > vth)).fill(tref); //keep neurons that fired in refractory period
    v.elem(find(v > vth)).fill(vreset); //and reset them
    
    //FORCE
    if (i%trainStep == 0 && i > trainStart && i < trainStop)
    {
      err = wOut*r - tgt.col(i); //error
      P = P - (P*r*r.t()*P)/(1 + arma::as_scalar(r.t()*P*r)); //update P
      wOut = wOut - err*r.t()*P.t(); //update output weights
      w = G*net.w + Q*wFb*wOut; //update weight matrix
    
      cout << "Iteration: " << i << "\t Error: " << err << endl;
    }
  }
  
  //Save new values
  net.v = v;
  net.r = r;
  net.h = h;

  ostringstream Nstream, Gstream, Qstream;
  Nstream << N;
  Qstream << Q;
  Gstream << G;

  //P.save("P.dat", raw_ascii);
  //vSave.save("v.dat", raw_ascii);
  //rSave.save("r_N_" + Nstream.str() + "_G_" + Gstream.str() + "_Q_" + Qstream.str() + ".dat", raw_ascii);
  rSave.save("r_disc.dat", raw_ascii);
  //hSave.save("h.dat", raw_ascii);
  //wOutSave.save("wOut_N_" + Nstream.str() + "_G_" + Gstream.str() + "_Q_" + Qstream.str() + ".dat", raw_ascii);
  wOutSave.save("wOut_disc.dat", raw_ascii);
  //spikes.save("spikes.dat", raw_ascii);
  //w.save("w.dat", raw_ascii);
}
#endif
