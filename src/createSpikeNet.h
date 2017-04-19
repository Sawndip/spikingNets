// Automatically translated using m2cpp 0.5 on 2017-02-27 21:10:46
            
#ifndef CREATESPIKENET_M_HPP
#define CREATESPIKENET_M_HPP

#include "mconvert.h"
#include <armadillo>
#include <cmath>
using namespace arma ;

struct _Net
{
  int N, nIn, nOut;
  float vth, vreset, vinf, tref, tm, td, tr, p, G, Q;
  vec v, r, h;
  mat w, wIn, wOut, wFb;
} ;

_Net createSpikeNet(float vth, float vreset, float vinf, float tref, float tm, float td, float tr, //LIF parameters
                    int N, float p, int nIn, int nOut, float G, float Q, mat w, mat wIn, mat wOut, mat wFb, //network parameters
                    vec v, vec r, vec h) //dynamical variables
{
  _Net net ;

  //LIF parameters
  net.vth = vth; //threshold voltage
  net.vreset = vreset; //reset voltage
  net.vinf = vinf; //bias current
  net.tref = tref; //refractory period
  net.tm = tm; //membrane relaxation time
  net.td = td; //synaptic decay time
  net.tr = tr; //synaptic rise time
  
  //Network parameters
  net.N = N; //number of neurons
  net.p = p; //network sparsity
  net.nIn = nIn; //number of input channels
  net.nOut = nOut; //number of output channels
  net.G = G ; //coupling of the weight matrix to the initial random weights
  net.Q = Q ; //coupling of the weight matrix to the learned weights
  
  //Initial values
  net.w = w; //initial random weights
  net.wIn = wIn; //initial random input weights
  net.wOut = wOut; //decoders
  net.wFb = wFb; //encoders
  
  net.v = v; //initial voltages
  net.r = r; //initial firing rates
  net.h = h; //initial somethings
 
  return net;
}
#endif
