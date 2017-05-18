#include <armadillo>
#include <fstream>
#include <iostream>
using namespace std;
using namespace arma;

int main()
{
  mat P, Pnew, dP, num, denmat;
  vec r;
  float den;

  P.load("/home/neurociencia/testP/P.try");
  r.load("/home/neurociencia/testP/r.try");

  num = P*r*r.t()*P;
  denmat = r.t()*P*r;
  den = 1 + as_scalar(denmat);
  dP = num/den;
  Pnew = P - dP;

  cout << "Numerator: \n";
  cout << num << endl;
  cout << "Den matrix \n";
  cout << denmat << endl;
  cout << "Denominator: \n";
  cout << den << endl;
  cout << "dP: \n";
  cout << dP << endl;
  cout << "New P: \n";
  cout << Pnew << endl;

  return 0;
}
