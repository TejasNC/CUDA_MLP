#include "../include/module.hpp"

#include "../include/layer_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <random>

extern cublasHandle_t global_cublas_handle;

Linear::Linear(int m, int n, int b, cudaStream_t stream)
    : m(m), n(n), b(b), W(m, n, stream), W0(n, 1, stream), dW(m, n, stream), dW0(n, 1, stream),
      Z(n, b, stream), stream(stream), A(m, b, stream), dLdA(m, b, stream)
{
    // Initialize weights with Xavier initialization
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / (m + n)));
    std::vector<float>              h_W(m * n);
    std::generate(h_W.begin(), h_W.end(), [&]() { return dist(gen); });
    W.copy_from_host(h_W);
    W0.zero(); // Initialize bias to zero
}

const Tensor& Linear::forward(const Tensor& input_)
{
    assert(input_.rows() == m && input_.cols() == b);

    A.copy_from(input_);
    // Perform matrix multiplication: Z = W^T * A + W0 (column-major layout)
    float alpha = 1.0f;
    float beta  = 0.0f;
    cublasSetStream(global_cublas_handle, stream);

    cublasSgemm(global_cublas_handle,
                CUBLAS_OP_T,                               // W^T
                CUBLAS_OP_N, n, b, m, &alpha, W.data(), m, // Leading dimension = m
                A.data(), m,                               // Leading dimension = m
                &beta, Z.data(), n                         // Leading dimension = n
    );

    launch_bias_add_kernel(Z.data(), W0.data(), n, b, stream);

    return Z;
}

const Tensor& Linear::backward(const Tensor& dLdZ)
{

    assert(dLdZ.rows() == n && dLdZ.cols() == b);

    // Compute dLdA =  W @ dLdZ
    float alpha = 1.0f;
    float beta  = 0.0f;

    cublasSetStream(global_cublas_handle, stream);
    cublasSgemm(global_cublas_handle,
                CUBLAS_OP_N, // No transpose
                CUBLAS_OP_N, m, b, n, &alpha, W.data(), m, dLdZ.data(), n, &beta, dLdA.data(), m);

    // Compute dW = A @ dLdZ^T
    cublasSgemm(global_cublas_handle,
                CUBLAS_OP_N, // No transpose
                CUBLAS_OP_T, m, n, b, &alpha, A.data(), m, dLdZ.data(), n, &beta, dW.data(), m);

    // Compute dW0 = sum(dLdZ, axis=1)
    launch_bias_grad_kernel(dLdZ.data(), dW0.data(), n, b, stream);

    return dLdA;
}

void Linear::update(float learning_rate)
{
    // Update weights: W -= learning_rate * dW
    float alpha = -learning_rate;
    float beta  = 1.0f;

    cublasSetStream(global_cublas_handle, stream);
    cublasSaxpy(global_cublas_handle, m * n, &alpha, dW.data(), 1, W.data(), 1);

    // Update bias: W0 -= learning_rate * dW0
    cublasSaxpy(global_cublas_handle, n, &alpha, dW0.data(), 1, W0.data(), 1);
}

ReLU::ReLU(int n, int b, cudaStream_t stream)
    : n(n), b(b), A(n, b, stream), dLdZ(n, b, stream), mask(n, b, stream), stream(stream)
{
}

const Tensor& ReLU::forward(const Tensor& Z)
{
    assert(Z.rows() == n && Z.cols() == b);
    launch_relu_forward_kernel(Z.data(), A.data(), mask.data(), n, b, stream);
    return A;
}

const Tensor& ReLU::backward(const Tensor& dLdA)
{
    assert(dLdA.rows() == n && dLdA.cols() == b);
    launch_relu_backward_kernel(dLdA.data(), mask.data(), dLdZ.data(), n, b, stream);
    return dLdZ;
}

SoftMax::SoftMax(int n, int b, cudaStream_t stream)
    : n(n), b(b), A(n, b, stream), dLdZ(n, b, stream), stream(stream)
{
}

const Tensor& SoftMax::forward(const Tensor& Z)
{
    assert(Z.rows() == n && Z.cols() == b);
    launch_softmax_forward_kernel(Z.data(), A.data(), n, b, stream);
    return A;
}

const Tensor& SoftMax::backward(const Tensor& dLdA)
{
    assert(dLdA.rows() == n && dLdA.cols() == b);
    // copy dLdA to dLdZ
    cudaMemcpyAsync(dLdZ.data(), dLdA.data(), n * b * sizeof(float), cudaMemcpyDeviceToDevice,
                    stream);
    cudaStreamSynchronize(stream);

    // don't need to apply softmax gradient here, as it is already done in NLL loss
    return dLdZ;
}

CrossEntropyLoss::CrossEntropyLoss(int n, int b, cudaStream_t stream)
    : n(n), b(b), dLdA(n, b, stream), stream(stream)
{
}

float CrossEntropyLoss::calc_loss(const Tensor& A, const Tensor& target)
{
    assert(A.rows() == n && A.cols() == b);
    assert(target.rows() == 1 && target.cols() == b);

    // Compute the cross-entropy loss
    float loss = 0.0f;
    launch_cross_entropy_loss_kernel(A.data(), target.data(), &loss, n, b, stream);
    return loss / (float) b; // Average loss over batch
}

const Tensor& CrossEntropyLoss::calc_grad(const Tensor& A, const Tensor& target)
{
    assert(A.rows() == n && A.cols() == b);
    assert(target.rows() == 1 && target.cols() == b);

    // Compute the gradient of cross-entropy loss w.r.t. softmax output
    launch_cross_entropy_grad_kernel(A.data(), target.data(), dLdA.data(), n, b, stream);
    return dLdA;
}

Tensor CrossEntropyLoss::predict_class(const Tensor& A)
{
    assert(A.rows() == n && A.cols() == b);

    Tensor predictions(1, b, stream);
    launch_predict_class_kernel(A.data(), predictions.data(), n, b, stream);
    return predictions;
}
