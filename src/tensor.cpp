#include "../include/tensor.hpp"

#include "../include/cuda_utils.hpp"
#include "../include/tensor_kernels.hpp"

#include <cassert>

Tensor::Tensor(int rows, int cols, cudaStream_t stream)
    : d_data(nullptr), stream(stream), _rows(rows), _cols(cols)
{
    assert(rows > 0 && cols > 0);
    allocate();
}

Tensor::Tensor(const std::vector<float>& host_data, int rows, int cols, cudaStream_t stream)
    : d_data(nullptr), stream(stream), _rows(rows), _cols(cols)
{
    assert(rows > 0 && cols > 0);
    allocate();
    copy_from_host(host_data);
}

Tensor::Tensor(int rows, int cols, float value, cudaStream_t stream)
    : d_data(nullptr), stream(stream), _rows(rows), _cols(cols)
{
    assert(rows > 0 && cols > 0);
    allocate();
    fill(value);
}

Tensor::~Tensor() { free(); }

Tensor::Tensor(Tensor&& other) noexcept
    : d_data(other.d_data), stream(other.stream), _rows(other._rows), _cols(other._cols)
{
    other.d_data = nullptr;
    other.stream = nullptr;
    other._rows  = 0;
    other._cols  = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept // uses move semantics
{
    if (this != &other)
    {
        free();
        d_data = other.d_data;
        // stream = other.stream;
        stream = 0; // Reset stream to default, as we cannot move streams (check this)
        _rows  = other._rows;
        _cols  = other._cols;

        other.d_data = nullptr;
        other.stream = nullptr;
        other._rows  = 0;
        other._cols  = 0;
    }
    return *this;
}

std::vector<float> Tensor::to_cpu() const
{
    std::vector<float> host_data(size());
    CUDA_CHECK(
        cudaMemcpyAsync(host_data.data(), d_data, memory_size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host_data;
}

void Tensor::copy_from_host(const std::vector<float>& host_data)
{
    assert(host_data.size() == size());
    CUDA_CHECK(
        cudaMemcpyAsync(d_data, host_data.data(), memory_size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaStreamSynchronize(stream)); // TODO: remove this add add an explicit synchronize call
}

void Tensor::copy_from(const Tensor& other)
{
    assert(is_valid() && other.is_valid());
    assert(_rows == other._rows && _cols == other._cols);
    CUDA_CHECK(
        cudaMemcpyAsync(d_data, other.d_data, memory_size(), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void Tensor::zero() { fill(0.0f); }

void Tensor::fill(float value)
{
    assert(is_valid());
    launch_fill_kernel(d_data, value, size(), stream);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Tensor::allocate()
{
    if (is_valid())
        return; // Already allocated
    CUDA_CHECK(cudaMalloc(&d_data, memory_size()));
}

void Tensor::free()
{
    if (d_data)
    {
        CUDA_CHECK(cudaFree(d_data));
        d_data = nullptr;
    }
}
