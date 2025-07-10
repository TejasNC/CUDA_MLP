#pragma once

#include <cuda_runtime.h>

// Tensor kernel function declarations
void launch_fill_kernel(float* d_data, float value, int size, cudaStream_t stream);
