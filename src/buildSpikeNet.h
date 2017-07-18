#include <armadillo> /* matrix algebra */
#include <sstream> /* stringstream */
#include <fstream> /* (i|o)fstream */
#include <iostream> /* cout */
#include <sys/stat.h> /* stat */
using namespace std;
using namespace arma;

template <typename T>
string toString(T value)
{
  ostringstream stream;
  stream << value;
  return stream.str();
}

/* Declare basic _Net structure */
struct _Net
{
  int N, nIn, nOut;
  float vth, vreset, vinf, tref, tm, td, tr, p, G, Q, lambda;
  vec v, r, h;
  mat w0, wIn, wOut, wFb;
};

/* Create a new _Net from within a script/function - mostly deprecated in
 * favour of loadSpikeNet() */
_Net createSpikeNet(float vth, float vreset, float vinf, float tref, float tm, float td, float tr, int N, float p, int nIn, int nOut, float G, float Q, float lambda, mat w0, mat wIn, mat wOut, mat wFb, vec v, vec r, vec h)
{
  _Net myNet;

  /* LIF parameters */
  myNet.vth = vth; //threshold voltage
  myNet.vreset = vreset; //reset voltage
  myNet.vinf = vinf; //bias current
  myNet.tref = tref; //refractory period
  myNet.tm = tm; //membrane relaxation time
  myNet.td = td; //synaptic decay time
  myNet.tr = tr; //synaptic rise time
  
  /* Network parameters */
  myNet.N = N; //number of neurons
  myNet.p = p; //network sparsity
  myNet.nIn = nIn; //number of input channels
  myNet.nOut = nOut; //number of output channels
  myNet.G = G; //coupling of the weight matrix to the initial random weights
  myNet.Q = Q; //coupling of the weight matrix to the learned weights
  myNet.lambda = lambda; //training rate

  /* Initial values */
  myNet.w0 = w0; //initial random weights
  myNet.wIn = wIn; //initial random input weights
  myNet.wOut = wOut; //decoders
  myNet.wFb = wFb; //encoders
  
  myNet.v = v; //initial voltages
  myNet.r = r; //initial firing rates
  myNet.h = h; //initial somethings
 
  return myNet;
};

/* Load _Net object from path netPath containing all the parameters in its
 * subdirectories - note that netPath should be an absolute path */
_Net loadSpikeNet(string netPath, string initPath)
{
  _Net myNet;
  
  /* Sanity checks on netPath's subdirectory structure */
  struct stat st;

  if (stat((netPath + toString("static/")).c_str(), &st) == 0)
  {
    /* Load LIF parameters - voltage and decay time constants */

    ifstream LIF (realpath((netPath + toString("static/LIF.dat")).c_str(), NULL));

    if (LIF.is_open())
    {
      LIF >> myNet.vth;
      LIF >> myNet.vreset;
      LIF >> myNet.vinf;
      LIF >> myNet.tref;
      LIF >> myNet.tm;
      LIF >> myNet.td;
      LIF >> myNet.tr;
    }
    else
    {
      cout << "error opening LIF file\n";
    }

    LIF.close();

    /* Load network architecture - # neurons, sparsity, # I/O, couplings */
    ifstream arch (realpath((netPath + toString("static/arch.dat")).c_str(), NULL));

    if (arch.is_open())
    {
      arch >> myNet.N;
      arch >> myNet.p;
      arch >> myNet.nIn;
      arch >> myNet.nOut;
      arch >> myNet.G;
      arch >> myNet.Q;
      arch >> myNet.lambda;
    }
    else
    {
      cout << "error opening arch file\n";
    }

    arch.close();

    /* Load network weights - except wOut */
    myNet.w0.load(realpath((netPath + "static/w0.dat").c_str(), NULL), raw_ascii);
    myNet.wIn.load(realpath((netPath + "static/wIn.dat").c_str(), NULL), raw_ascii);
    myNet.wFb.load(realpath((netPath + "static/wFb.dat").c_str(), NULL), raw_ascii);
    

  }
  else
  {
    cout << "please create /static directroy\n";
  }

  /* Load network state in phase space plus P and wOut */

  if (stat(initPath.c_str(), &st) == 0)
  {
    myNet.wOut.load(realpath((initPath + "wOut.dat").c_str(), NULL), raw_ascii);
    myNet.v.load(realpath((initPath + toString("v.dat")).c_str(), NULL), raw_ascii);
    myNet.r.load(realpath((initPath + toString("r.dat")).c_str(), NULL), raw_ascii);
    myNet.h.load(realpath((initPath + toString("h.dat")).c_str(), NULL), raw_ascii);
  } else cout << "error - specified initPath does not exist\n";

  return myNet;
}
