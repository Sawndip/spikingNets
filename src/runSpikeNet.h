#include "buildSpikeNet.h"
#include <string>
#include <cstdlib>
using namespace std;
using namespace arma;

_Net runSpikeNet(_Net net, mat input, mat tgt, float dt, int trainStep, int saveRate, int numTrial, string netPath, string savePath, bool spikeTest, bool rasterPlot, ofstream& logfile)
{
  // Shouldn't need this: arma_rng::set_seed(42); 
  // note that trainStep should be given in units of dt, e.g. if integration
  // time is dt = 0.001 seconds and the training occurs every 0.05 seconds,
  // then int trainStep = 50;
  // Setting int trainStep = INFINITY the network runs without training, so
  // this function can be used to initialize it too.
  
  int T = input.n_cols; //time duration of input
  
  // LIF neurons' parameters
  float vth = net.vth;
  float vreset = net.vreset;
  float vinf = net.vinf;
  float tref = net.tref;
  float tm = net.tm;
  float td = net.td;
  float tr = net.tr;

  // Network parameters
  int N = net.N;
  float p = net.p;
  int nIn = net.nIn;
  int nOut = net.nOut;
  float G = net.G;
  float Q = net.Q;
  float lambda = net.lambda;
  cout << "loaded LIF and arch params" << endl;  
  // Weights
  mat wIn = net.wIn;
  mat wOut = net.wOut;
  mat wFb = net.wFb;
  mat w0 = net.w0;
  cout << "loaded weights" << endl;
  // Iinitial values
  vec v = net.v;
  vec r = net.r;
  vec h = net.h;
  vec dv, dr, dh, spikes, ref;
  ref.zeros(N);
  
  cout << "loaded init vals" << endl;
  /* FORCE initial values */
  mat P, err;
  P.load(netPath + "init/P.dat", raw_ascii);
  cout << "loaded force" << endl;
  /* Save time evolution of firing rate */
  mat rSave = r;
  mat raster;
  cout << "time evol defined" << endl;
  /* Test for chaos deleting spike */
  bool spikeDeleted = false;
  int deleteTime = (int)(T/2.0); //delete a spike at t = totalTime/2


  /* Integration loop */
  vec inputNow, tgtNow;
  
  for (int i=0; i<T; i=i+1)
  {

    inputNow = input.col(i);
    tgtNow = tgt.col(i);


    if (i%500 == 0)
    {
      logfile << "Iteration " << i << " out of " << T << "\t" << (float)(100*i/T) << " \% progress \n";
    }


    /* LIF equations */

    // Voltage ODE
    dv = (-v + vinf + G*w0*r + Q*wFb*wOut*r + wIn*inputNow)/tm; //voltage
    
    // Update neurons not in refractory period
    v.elem(find(ref <= 0)) = v.elem(find(ref <= 0)) + dv.elem(find(ref <= 0))*dt;
   
    // Double exponential filter
    dh = -h/tr;
    h = h + dh*dt + conv_to<vec>::from(v > vth)/(tr*td);

    dr = -r/td + h;
    r = r + dt*dr;

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

    if (i%trainStep == 0 && i > 0)
    {
      err = wOut*r - tgtNow; //error
      P = P - (P*r*r.t()*P)/(1 + as_scalar(r.t()*P*r)); //update P
      wOut = wOut - err*r.t()*P.t(); //update output weights
    
      if (i%500 == 0)
      {
        logfile << "FORCING w/ error: " << arma::as_scalar(accu(err)) << "\n";
      }
    }


    /* Save time evolution of variables */
    
    if (i%saveRate == 0)
    {
      rSave = join_rows(rSave, r);
    }

    if (rasterPlot)
    {
      raster = join_rows(raster, spikes);
    }
  } //ENDFOR
  
  /* Save final values as initial values for next trial */
  v.save(netPath + toString("init/v.dat"), raw_ascii);
  h.save(netPath + toString("init/h.dat"), raw_ascii);
  r.save(netPath + toString("init/r.dat"), raw_ascii);
  P.save(netPath + toString("init/P.dat"), raw_ascii);
  wOut.save(netPath + toString("init/wOut.dat"), raw_ascii);

  /* Save fire rate and final weights to get output */
  string saveData;

  if (numTrial < 10)
  {
    saveData = "0" + toString(numTrial);
  }
  else
  {
    saveData = toString(numTrial);
  }
  
  rSave.save(savePath + "r" + saveData + ".dat", raw_ascii);
  wOut.save(savePath + "wOut" + saveData + ".dat", raw_ascii);

  _Net newNet = createSpikeNet(vth, vreset, vinf, tref, tm, td, tr, N, p, nIn, nOut, G, Q, lambda, w0, wIn, wOut, wFb, v, r, h);

  return newNet;
};

_Net equilibrateSpikeNet(_Net myNet, float dt, float time, string netPath, string savePath, ofstream& logfile)
{
  int T = round(time/dt);
  mat dummyInp, dummyTgt;
  dummyInp.zeros(myNet.nIn, T);
  dummyTgt.zeros(myNet.nOut, T);

  int trainStep = (int)INFINITY;
  int saveRate = 100;
  bool spikeTest = false;
  bool rasterPlot = false;
  int numTrial = 0;

  _Net newNet = runSpikeNet(myNet, dummyInp, dummyTgt, dt, trainStep, saveRate, numTrial, netPath, savePath, spikeTest, rasterPlot, logfile);

  return newNet;
}
