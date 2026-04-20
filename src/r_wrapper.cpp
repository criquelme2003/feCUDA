#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
int suma(int x, int y) {
  return x + y;
}