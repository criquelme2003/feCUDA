install.packages("Rcpp")
Rcpp::sourceCpp("/workspace/src/r_wrapper.cpp")

result <- suma(2,4)
result
