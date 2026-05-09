#include <Rcpp.h>
#include "../include/core/rcpp.cuh"







// [[Rcpp::export]]
Rcpp::List test_tensor(Rcpp::NumericMatrix mat)
{
    RcppTensorResult<float> tensor(
        mat,
        MemorySpace::Device,
        RcppTensorResult<float>::Layout::RowMajor
    );

    return Rcpp::List::create(
        Rcpp::Named("batch") = tensor.getBatch(),
        Rcpp::Named("m") = tensor.getM(),
        Rcpp::Named("n") = tensor.getN(),
        Rcpp::Named("k") = tensor.getK(),
        Rcpp::Named("is_device") = tensor.isDevicePtr()
    );
}