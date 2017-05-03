#include "buildSpikeNet.h"
#include <string>
#include <cstdlib>
using namespace std;
using namespace arma;

_Net runSpikeNet(_Net net, mat input, mat tgt, float dt, int trainStep, int trainStart, int trainStop, int saveRate, int saveData, int saveFORCE, string savePath, bool spikeTest)
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
  float p = net.p;
  int nIn = net.nIn;
  int nOut = net.nOut;
  float G = net.G;
  float Q = net.Q;
  float lambda = net.lambda;
  
  mat wIn = net.wIn;
  mat wOut = net.wOut;
  mat wFb = net.wFb;
  mat w0 = net.w0;
 
  //Network initial values
  vec v = net.v;
  vec r = net.r;
  vec h = net.h;
  vec dv = net.dv;
  vec dr = net.dr;
  vec dh = net.dh;
  vec spikes = net.spikes;
  vec ref = net.ref;
  
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

  vec inputNow, tgtNow;

  for (int i=0; i<T; i=i+1)
  {

    inputNow = input.col(i);
    tgtNow = tgt.col(i);

    //Keep track of variable evolution every saveRate steps plus final
    //iteration
    if ((i%saveRate == 0 || i == T-1) && i > 0)
    {
      rSave = join_rows(rSave, r);
      vSave = join_rows(vSave, v);
    }

    if (i%500 == 0)
    {
      cout << "Iteration " << i << " out of " << T << "\t" << (float)(100*i/T) << " \% progress" << endl;
    }


    /* LIF equations */

    // Voltage ODE
    dv = (-v + vinf + G*w0*r + Q*wFb*wOut*r + wIn*inputNow)/tm; //voltage
    // Update neurons not in refractory period
    v.elem(find(ref <= 0)) = v.elem(find(ref <= 0)) + dv.elem(find(ref <= 0))*dt;
   
    // Double exponential filter
    dr = -r/td + h;
    r = r + dt*dr;

    dh = -h/tr;
    h = h + dh*dt + conv_to<vec>::from(v > vth)/(tr*td);

    /* Spike deletion - only for testing chaos */
    if (spikeTest && !spikeDeleted && (i == deleteTime))
    {
      uvec deletedNeuron = find(v > vth, 1, "first"); //
      v.elem(deletedNeuron).fill(vreset);
      spikeDeleted = true;
    }

    /* Fire prescription */
    spikes = (conv_to<vec>::from(v > vth)); //keep track of spikes before updating
    ref.elem(find(ref > 0)) = ref.elem(find(ref > 0)) - dt; //subtract the elapsed timestep from neurons in refractory period
    ref.elem(find(v > vth)).fill(tref); //keep neurons that fired in refractory period
    v.elem(find(v > vth)).fill(vreset); //and reset them
    

    /* FORCE learning */

    if (i%trainStep == 0 && i > trainStart && i < trainStop)
    {
      err = wOut*r - tgtNow; //error
      P = P - (P*r*r.t()*P)/(1 + as_scalar(r.t()*P*r)); //update P
      wOut = wOut - err*r.t()*P.t(); //update output weights
    
      if (i%500 == 0)
      {
        cout << "FORCING w/ error: " << arma::as_scalar(accu(err)) << endl;
      }

      if ((i%saveFORCE == 0 || i == T - 1) && i > 0)
      {
        P.save(savePath + "P" + toString(i) + ".dat", raw_ascii);
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
  wOut.save(savePath + "wOut" + toString(saveData) + ".dat", raw_ascii);

  _Net newNet = createSpikeNet(vth, vreset, vinf, tref, tm, td, tr, N, p, nIn, nOut, G, Q, lambda, w0, wIn, wOut, wFb, v, r, h, dv, dr, dh, spikes, ref, P, err);

  return newNet;
};

_Net equilibrateSpikeNet(_Net myNet, float dt, float time)
{
  int T = round(time/dt);
  mat dummyInp, dummyTgt;
  dummyInp.zeros(myNet.nIn, T);
  dummyTgt.zeros(myNet.nOut, T);

  int trainStep = (int)INFINITY;
  int trainStart = 0;
  int trainStop = T;
  int saveRate = (int)INFINITY;
  int saveFORCE = (int)INFINITY;
  bool spikeTest = false;
  int saveData = 0;
  string savePath = "none";

  _Net newNet = runSpikeNet(myNet, dummyInp, dummyTgt, dt, trainStep, trainStart, trainStop, saveRate, saveData, saveFORCE, savePath, spikeTest);

  return newNet;
}
