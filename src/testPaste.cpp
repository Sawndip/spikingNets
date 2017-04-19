#define INFINITY
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>
using namespace std;

int main()
{
  //just a sstream tutorial - irrelevant if you already know how to use it :-)
  
  int N = 42;
  float Q = 0.05;
  string b = "b";
  ostringstream convert;
  //convert << N;
  string s = "a" + b + convert.str();
  ostringstream convertQ;
  convertQ << Q;
  s = s + convertQ.str();

  for (int i=0; i<N; i++)
  {
    convert << i;
    cout << convert.str() << endl;
  }
  return 0;
}
