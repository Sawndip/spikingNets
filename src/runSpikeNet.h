// Automatically translated using m2cpp 0.5 on 2017-02-27 19:06:26
//
// This function takes an already created network (_Net object) and integrates
// its dynamics with timestep dt given a certain input.

#ifndef RUNSPIKENET_M_HPP
#define RUNSPIKENET_M_HPP

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

template <typename T>
string toString(T value)
{
  ostringstream stream;
  stream << value;
  return stream.str();
}

_Net runSpikeNet(_Net net, mat input, mat tgt, float dt, int trainStep, int trainStart, int trainStop, float trainRate, int saveRate, int saveData, int saveFORCE, string savePath, bool spikeTest, int discTask)
{
  arma_rng::set_seed(42); 
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
  int nIn = net.nIn;
  int nOut = net.nOut;
  float G = net.G;
  float Q = net.Q;
  float p = net.p;

  //Network initial values
  vec v = net.v;
  vec r = net.r;
  vec h = net.h;
  vec dv = net.dv;
  vec dr = net.dr;
  vec dh = net.dh;
  mat spikes = net.spikes;
  mat ref = net.ref;
  mat wIn = net.wIn;
  mat wOut = net.wOut;
  mat wFb = net.wFb;
  mat w = G*net.w + Q*wFb*wOut;
  int T = input.n_cols; //time duration of input
  
  //FORCE initial values
  mat P = net.P;
  mat err = net.err;
  
  //Save time evolution of variables here
  mat vSave = v;
  mat rSave = r;
  mat hSave = h;
  mat wOutSave = wOut;

  //Test for chaos deleting spike
  bool spikeDeleted = false;
  int deleteTime = (int)(T/2.0); //delete a spike only in the second half of the interval

  //Utils
  //P.save("/home/neurociencia/utils/P0.dat", raw_ascii);
  w.save(savePath + "w_init.dat", raw_ascii);
  wFb.save(savePath + "wFb_init.dat", raw_ascii);
  wIn.save(savePath + "wIn_init.dat", raw_ascii);

  //Discrimination task
  vec inputNow, tgtNow;

  for (int i=0; i<T; i=i+1)
  {

    if (discTask == 0)
    {
      inputNow = input.col(i);
      tgtNow = tgt.col(i);
    }
    else if (discTask == 1)
    {
      inputNow << (float) (i < 5000) << endr
               << (float) (i > 20000)*(i < 40000) + (i > 60000)*(i < 80000) << endr
               << (float) 0 << endr;
      tgtNow   << (float) (i > 100000) << endr
               << (float) 0 << endr;
    }
    else if (discTask == 2)
    {
      inputNow << (float) (i < 5000) << endr
               << (float) (i < 20000)*(i < 40000) << endr
               << (float) (i > 60000)*(i < 80000) << endr;
      tgtNow   << (float) 0 << endr
               << (float) (i > 100000) << endr;
    }
    else if (discTask == 3)
    {
      inputNow << (float) (i < 5000) << endr
               << (float) (i > 60000)*(i < 80000) << endr
               << (float) (i > 20000)*(i < 40000) << endr;
      tgtNow   << (float) 0 << endr
               << (float) (i > 100000) << endr;
    }
    else if (discTask == 4)
    {
      inputNow << (float) (i < 5000) << endr
               << (float) 0 << endr
               << (float) (i > 20000)*(i < 40000) + (i > 60000)*(i < 80000) << endr;
      tgtNow   << (float) (i > 100000) << endr
               << (float) 0 << endr;
    }

    //Keep track of variable evolution every 100 steps
    if (i%saveRate == 0 && i > 0)
    {
      rSave = join_rows(rSave, r);
      vSave = join_rows(vSave, v);
    }
    
    if (i%500 == 0)
    {
      cout << "Iteration " << i << " out of " << T << "\t" << (float)(100*i/T) << " \% progress" << endl;
    }
    
    //LIF EDOs system
    dv = (-v + vinf + w*r + wIn*inputNow)/tm; //voltage
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
    spikes.row(i) = (conv_to<vec>::from(v > vth)).t(); //keep track of spikes before updating
    ref.elem(find(ref > 0)) = ref.elem(find(ref > 0)) - dt; //subtract the elapsed timestep from neurons in refractory period
    ref.elem(find(v > vth)).fill(tref); //keep neurons that fired in refractory period
    v.elem(find(v > vth)).fill(vreset); //and reset them
    
    //FORCE
    if (i%trainStep == 0 && i > trainStart && i < trainStop)
    {
      err = wOut*r - tgtNow; //error
      P = P - (P*r*r.t()*P)/(1 + arma::as_scalar(r.t()*P*r)); //update P
      wOut = wOut - err*r.t()*P.t(); //update output weights
      w = G*net.w + Q*wFb*wOut; //update weight matrix
    
      if (i%500 == 0)
      {
        cout << "FORCING w/ error: " << arma::as_scalar(accu(err)) << endl;
      }

      if (i%saveFORCE == 0)
      {
        P.save(savePath + "P" + toString(i) + ".dat", raw_ascii);
        w.save(savePath + "w" + toString(i) + ".dat", raw_ascii);
        wOut.save(savePath + "wOut" + toString(i) + ".dat", raw_ascii);
      }
    }
  } //ENDFOR
  
  //Save new values
  net.v = v;
  net.r = r;
  net.h = h;

  rSave.save(savePath + "r" + toString(saveData) + ".dat", raw_ascii);
  vSave.save(savePath + "v" + toString(saveData) + ".dat", raw_ascii);
  w.save(savePath + "w" + toString(saveData) + ".dat", raw_ascii);
  wOut.save(savePath + "wOut" + toString(saveData) + ".dat", raw_ascii);

  _Net newNet = createSpikeNet(vth, vreset, vinf, tref, tm, td, tr, N, p, nIn, nOut, G, Q, w, wIn, wOut, wFb, v, r, h, dv, dr, dh, spikes, ref, P, err);

  return newNet;
}

#endif
