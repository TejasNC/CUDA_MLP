#include "../include/cuda_utils.hpp"

// CUDA kernel to fill a tensor with a specific value
// This kernel is used to initialize tensors with a constant value.
__global__ void fill_kernel(float *data, float value, int n) {
  int idx = cuda_utils::get_1d_idx();
  if (idx < n) {
    data[idx] = value;
  }
}

void launch_fill_kernel(float *d_data, float value, int size,
                        cudaStream_t stream) {

  dim3 block_size = (256);
  dim3 grid_size = CEIL_DIV(size, block_size.x);

  // Launch the kernel
  fill_kernel<<<grid_size, block_size, 0, stream>>>(d_data, value, size);

  // Check for errors in kernel launch
  CUDA_CHECK_KERNEL();
}
