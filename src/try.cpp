#include <iostream> /* cout */
#include <fstream> /* (i|o)fstream */
#include <sstream> /* toString */
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

int main()
{
  float vth = -40.0;
  string netPath = "/home/neurociencia/";
  ofstream LIF ((netPath + toString("LIF.dat")).c_str());

  if (LIF.is_open())
  {
    LIF << vth << "\n";
  }
  else cout << "unable to open file" << endl;

  LIF.close();
  
  float vnew;

  ifstream readLIF ((netPath + toString("LIF.dat")).c_str());

  if (readLIF.is_open())
  {
    readLIF >> vnew;
  } else cout << "unable to read LIF file" << endl;

  readLIF.close();
  cout << vnew << endl;
  
  string line;
  vec myVec(2000);
  int i = 0;
  ifstream in ("/home/neurociencia/svm/wOut.dat");

  while (in >> ws && getline(in, line));
  in.close();
  
  istringstream myLine(line);

  while (!myLine.eof())
  {
    myLine >> myVec(i);
    i++;
  }

  for (int j=0; j<2000; j++)
  {
    cout << myVec(j) << endl;
  }

  return 0;
}
