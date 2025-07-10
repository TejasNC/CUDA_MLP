#pragma once

#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

/*
  Tensor class is a wrapper around a 2D array of floats in CUDA device memory.
  It stores data in column-major format (FORTRAN-style) for optimal CUBLAS performance.
  Memory layout: [col0_row0, col0_row1, col0_row2, col1_row0, col1_row1, ...]
*/

class Tensor
{

  public:
    Tensor(int rows, int cols, cudaStream_t stream = 0); // Uninitialized
    Tensor(const std::vector<float>& host_data, int rows, int cols,
           cudaStream_t stream = 0);                                  // Load from host
    Tensor(int rows, int cols, float value, cudaStream_t stream = 0); // Fill with value

    ~Tensor();

    // disable copy semantics
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;

    // enable move semantics
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Metadata
    int    rows() const { return _rows; }
    int    cols() const { return _cols; }
    int    size() const { return _rows * _cols; }
    float* data() const { return d_data; }

    // Check if tensor is valid
    bool is_valid() const { return d_data != nullptr && _rows > 0 && _cols > 0; }

    // Transfer
    std::vector<float> to_cpu() const; // Device → Host
    void               copy_from_host(const std::vector<float>& host_data);
    void               copy_from(const Tensor& other); // Device → Device

    // Initialization
    void zero();            // Fill with zeros
    void fill(float value); // Fill with value

    // Stream management
    void         set_stream(cudaStream_t new_stream) { stream = new_stream; }
    cudaStream_t get_stream() const { return stream; }

    std::size_t memory_size() const { return size() * sizeof(float); }

  private:
    float*       d_data; // Pointer to device memory
    cudaStream_t stream; // CUDA stream for asynchronous operations
    int          _rows;
    int          _cols;

    void allocate();
    void free();

    inline int get_index(int row, int col) const
    {
        assert(row >= 0 && row < _rows && col >= 0 && col < _cols);
        return col * _rows + row; // Column-major layout
    }
};
