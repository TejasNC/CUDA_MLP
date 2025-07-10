#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CEIL_DIV(x, y) (((x) + (y) -1) / (y))

// CUDA error checking macro
#define CUDA_CHECK(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            throw std::runtime_error("CUDA error at " + std::string(__FILE__) + ":" +              \
                                     std::to_string(__LINE__) + ": " +                             \
                                     std::string(cudaGetErrorString(err)));                        \
        }                                                                                          \
    } while (0)

// CUDA kernel launch error checking
#define CUDA_CHECK_KERNEL()                                                                        \
    do                                                                                             \
    {                                                                                              \
        CUDA_CHECK(cudaGetLastError());                                                            \
    } while (0)

// Common CUDA thread/block calculations
#define CUDA_1D_KERNEL_CONFIG(n, block_size)                                                       \
    dim3 block_size_1d(block_size);                                                                \
    dim3 grid_size_1d((n + block_size - 1) / block_size)

#define CUDA_2D_KERNEL_CONFIG(rows, cols, block_x, block_y)                                        \
    dim3 block_size_2d(block_x, block_y);                                                          \
    dim3 grid_size_2d((cols + block_x - 1) / block_x, (rows + block_y - 1) / block_y)

// Common block sizes
#define CUDA_BLOCK_SIZE_1D 256
#define CUDA_BLOCK_SIZE_2D_X 16
#define CUDA_BLOCK_SIZE_2D_Y 16

// Only include device functions when compiling CUDA code
#ifdef __CUDACC__
namespace cuda_utils
{
// Get thread index in 1D grid
__device__ inline int get_1d_idx() { return blockIdx.x * blockDim.x + threadIdx.x; }

// Get thread indices in 2D grid
__device__ inline int get_2d_row() { return blockIdx.y * blockDim.y + threadIdx.y; }

__device__ inline int get_2d_col() { return blockIdx.x * blockDim.x + threadIdx.x; }

// Convert 2D indices to 1D (column-major)
__device__ inline int get_2d_idx(int row, int col, int rows)
{
    return col * rows + row; // Column-major: columns are contiguous
}
} // namespace cuda_utils
#endif // __CUDACC__
