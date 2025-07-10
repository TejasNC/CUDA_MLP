#pragma once

#include <cuda_runtime.h>

// Layer kernel function declarations
void launch_bias_add_kernel(float* Z, const float* b, int n, int batch_size, cudaStream_t stream);
void launch_bias_grad_kernel(float* dLdZ, float* dW0, int n, int b, cudaStream_t stream);
void launch_relu_forward_kernel(const float* Z, float* A, float* mask, int n, int b,
                                cudaStream_t stream);
void launch_relu_backward_kernel(const float* dLdA, const float* mask, float* dLdZ, int n, int b,
                                 cudaStream_t stream);
void launch_softmax_forward_kernel(const float* Z, float* A, int n, int b, cudaStream_t stream);
void launch_cross_entropy_loss_kernel(const float* A, const float* target, float* loss, int n,
                                      int b, cudaStream_t stream);
void launch_cross_entropy_grad_kernel(const float* A, const float* target, float* dLdA, int n,
                                      int b, cudaStream_t stream);
void launch_predict_class_kernel(const float* A, float* predictions, int n, int b,
                                 cudaStream_t stream);
