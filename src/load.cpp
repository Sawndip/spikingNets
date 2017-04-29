#include <iostream>
#include <cstdlib>
#include <fstream>
using namespace std;

class _Net
{
  int N, nIn, nOut;
  float LIF[6];
  float vth, vreset, vinf, tref, tm, td, tr, p, G, Q;
  //mat w, wIn, wOut, wFb;
  //vec v, r, h;
  
  public:

  void load(const char* fileLIF)
  {
    ifstream params(fileLIF);
    float data;
    int i = 0;
    while(!params.eof())
    {
      params >> LIF[i];
      cout << LIF[i] << endl;
      i =+1;
    }
  }
};

int main(int argc, char* argv[])
{
  _Net myNet;
  myNet.load(argv[1]);
  return(0);
}
