#include <armadillo> /* matrix algebra */
#include <sstream> /* stringstream */
#include <fstream> /* (i|o)fstream */
#include <iostream> /* cout */
using namespace std;
using namespace arma;

template <typename T>
string toString(T value)
{
  ostringstream stream;
  stream << value;
  return stream.str();
}

vec getLastLine(string pathToFile, int sz)
{
  string line;
  vec myVec(sz);
  ifstream myFile(pathToFile.c_str());
  int i = 0;

  if (myFile.is_open())
  {
    while (myFile >> ws && getline(myFile, line)); //go to last line
    myFile.close();
  } else cout << "unable to open file" << endl;

  istringstream myLine(line);

  while (!myLine.eof())
  {
    myLine >> myVec(i);
    i++;
  }

  return myVec;
}

struct _Net
{
  int N, nIn, nOut;
  float vth, vreset, vinf, tref, tm, td, tr, p, G, Q, lambda;
  vec v, r, h;
  mat w0, wIn, wOut, wFb;
};

_Net createSpikeNet(float vth, float vreset, float vinf, float tref, float tm, float td, float tr, int N, float p, int nIn, int nOut, float G, float Q, float lambda, mat w0, mat wIn, mat wOut, mat wFb, vec v, vec r, vec h)
{
  _Net myNet;

  //LIF parameters
  myNet.vth = vth; //threshold voltage
  myNet.vreset = vreset; //reset voltage
  myNet.vinf = vinf; //bias current
  myNet.tref = tref; //refractory period
  myNet.tm = tm; //membrane relaxation time
  myNet.td = td; //synaptic decay time
  myNet.tr = tr; //synaptic rise time
  
  //Network parameters
  myNet.N = N; //number of neurons
  myNet.p = p; //network sparsity
  myNet.nIn = nIn; //number of input channels
  myNet.nOut = nOut; //number of output channels
  myNet.G = G; //coupling of the weight matrix to the initial random weights
  myNet.Q = Q; //coupling of the weight matrix to the learned weights
  myNet.lambda = lambda; //training rate

  //Initial values
  myNet.w0 = w0; //initial random weights
  myNet.wIn = wIn; //initial random input weights
  myNet.wOut = wOut; //decoders
  myNet.wFb = wFb; //encoders
  
  myNet.v = v; //initial voltages
  myNet.r = r; //initial firing rates
  myNet.h = h; //initial somethings
 
  return myNet;
};

_Net loadSpikeNet(string netPath, string initPath)
{
  _Net myNet;
  
  // Load LIF parameters - voltage and decay time constants

  ifstream LIF ((netPath + toString("static/LIF.dat")).c_str());

  if (LIF.is_open())
  {
    LIF >> myNet.vth;
    LIF >> myNet.vreset;
    LIF >> myNet.vinf;
    LIF >> myNet.tref;
    LIF >> myNet.tm;
    LIF >> myNet.td;
    LIF >> myNet.tr;
  } else cout << "error opening LIF file \n";

  LIF.close();


  // Load network architecture - # neurons, sparsity, # I/O, couplings

  ifstream arch ((netPath + toString("static/arch.dat")).c_str());

  if (arch.is_open())
  {
    arch >> myNet.N;
    arch >> myNet.p;
    arch >> myNet.nIn;
    arch >> myNet.nOut;
    arch >> myNet.G;
    arch >> myNet.Q;
    arch >> myNet.lambda;
  } else cout << "error opening arch file \n";

  arch.close();

  // Load network weights

  myNet.w0.load(netPath + "static/w0.dat", raw_ascii);
  myNet.wIn.load(netPath + "static/wIn.dat", raw_ascii);
  myNet.wFb.load(netPath + "static/wFb.dat", raw_ascii);
  
  myNet.wOut.load(initPath + "wOut.dat", raw_ascii);


  // Load network state in phase space

  myNet.v.load(initPath + toString("v.dat"), raw_ascii);
  myNet.r.load(initPath + toString("r.dat"), raw_ascii);
  myNet.h.load(initPath + toString("h.dat"), raw_ascii);
  
  return myNet;
}
