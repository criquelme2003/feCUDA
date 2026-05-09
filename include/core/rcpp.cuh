#include <Rcpp.h>
#include <cuda_runtime.h>
#include "types.cuh"

template <typename T = float>
class RcppTensorResult : public TensorResult<T>
{
  public:
    enum class Layout
    {
        RowMajor,
        ColumnMajor
    };

    RcppTensorResult(
        Rcpp::NumericMatrix x,
        MemorySpace target_space = MemorySpace::Host,
        Layout layout = Layout::RowMajor
    )
        : TensorResult<T>(
              MemorySpace::Host,
              1,              // batch
              x.nrow(),       // m
              x.ncol(),       // n
              1               // k
          )
    {
        int rows = x.nrow();
        int cols = x.ncol();

        T* host_data = this->getData();

        if (layout == Layout::RowMajor)
        {
            // R: column-major -> tensor: row-major
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    host_data[i * cols + j] = static_cast<T>(x(i, j));
                }
            }
        }
        else
        {
            // Mantiene layout column-major tal como viene desde R
            const double* r_data = x.begin();

            for (int idx = 0; idx < rows * cols; ++idx)
            {
                host_data[idx] = static_cast<T>(r_data[idx]);
            }
        }

        if (target_space == MemorySpace::Device)
        {
            this->move_to_device();
        }
    }
};