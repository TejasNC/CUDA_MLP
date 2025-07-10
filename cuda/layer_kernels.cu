#include "../include/cuda_utils.hpp"
#include <cfloat>

__global__ void bias_add_kernel(float *Z, const float *b, int n,
                                int batch_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < batch_size) {
    Z[cuda_utils::get_2d_idx(i, j, n)] += b[i];
  }

}

void launch_bias_add_kernel(float *Z, const float *b, int n, int batch_size,
                            cudaStream_t stream) {
  dim3 block(16, 16); // TODO: Adjust block size based on n and batch_size
  dim3 grid(CEIL_DIV(n, block.x), CEIL_DIV(batch_size, block.y));

  bias_add_kernel<<<grid, block, 0, stream>>>(Z, b, n, batch_size);

  CUDA_CHECK_KERNEL();
}

__global__ void bias_grad_kernel(float *dLdZ, float *dW0, int n, int b) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    float sum = 0.0f;
    for (int col = 0; col < b; ++col) {
      sum += dLdZ[cuda_utils::get_2d_idx(row, col, n)];
    }
    dW0[row] = sum; // Gradient for bias
  }
}

void launch_bias_grad_kernel(float *dLdZ, float *dW0, int n, int b,
                             cudaStream_t stream) {

  dim3 block(256);
  dim3 grid(CEIL_DIV(n, block.x));

  bias_grad_kernel<<<grid, block, 0, stream>>>(dLdZ, dW0, n, b);
  cudaStreamSynchronize(stream);

  CUDA_CHECK_KERNEL();
}

__global__ void relu_forward_kernel(const float *Z, float *A, float *mask,
                                    int n, int b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * b) {
    A[idx] = fmaxf(Z[idx], 0.0f);              // ReLU activation
    mask[idx] = (Z[idx] > 0.0f) ? 1.0f : 0.0f; // Store mask for backpropagation
  }
}

void launch_relu_forward_kernel(const float *Z, float *A, float *mask, int n,
                                int b, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(CEIL_DIV(n * b, block.x));

  relu_forward_kernel<<<grid, block, 0, stream>>>(Z, A, mask, n, b);
  cudaStreamSynchronize(stream);

  CUDA_CHECK_KERNEL();
}

__global__ void relu_backward_kernel(const float *dLdA, const float *mask,
                                     float *dLdZ, int n, int b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * b) {
    dLdZ[idx] = dLdA[idx] * mask[idx]; // Apply mask to gradient
  }
}

void launch_relu_backward_kernel(const float *dLdA, const float *mask,
                                 float *dLdZ, int n, int b,
                                 cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(CEIL_DIV(n * b, block.x));

  relu_backward_kernel<<<grid, block, 0, stream>>>(dLdA, mask, dLdZ, n, b);
  cudaStreamSynchronize(stream);

  CUDA_CHECK_KERNEL();
}

__global__ void softmax_forward_kernel(const float *Z, float *A, int n, int b) {
  int col = cuda_utils::get_1d_idx(); // Each thread handles one batch sample

  if (col < b) {
    // Find max value across all classes for this batch sample (for numerical
    // stability)
    float max_val = Z[cuda_utils::get_2d_idx(0, col, n)];
    for (int row = 1; row < n; ++row) {
      max_val = fmaxf(max_val, Z[cuda_utils::get_2d_idx(row, col, n)]);
    }

    // Compute exp and sum across all classes for this batch sample
    float sum = 0.0f;
    for (int row = 0; row < n; ++row) {
      A[cuda_utils::get_2d_idx(row, col, n)] =
          expf(Z[cuda_utils::get_2d_idx(row, col, n)] - max_val);
      sum += A[cuda_utils::get_2d_idx(row, col, n)];
    }

    // Normalize across all classes for this batch sample
    for (int row = 0; row < n; ++row) {
      A[cuda_utils::get_2d_idx(row, col, n)] /= sum;
    }
  }
}

void launch_softmax_forward_kernel(const float *Z, float *A, int n, int b,
                                   cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(CEIL_DIV(b, block.x)); // Grid size based on batch size

  softmax_forward_kernel<<<grid, block, 0, stream>>>(Z, A, n, b);
  cudaStreamSynchronize(stream);

  CUDA_CHECK_KERNEL();
}

__global__ void cross_entropy_loss_kernel(const float *A, const float *target,
                                          float *loss, int n, int b) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < b) {
    // target[idx] contains the class index for batch idx
    int target_class = (int)target[idx];
    if (target_class >= 0 && target_class < n) {
      float log_prob =
          logf(A[cuda_utils::get_2d_idx(target_class, idx, n)] + 1e-10f);
      atomicAdd(loss, -log_prob); // Accumulate loss across batches
    }
  }
}

void launch_cross_entropy_loss_kernel(const float *A, const float *target,
                                      float *loss, int n, int b,
                                      cudaStream_t stream) {
  // Initialize loss to 0
  cudaMemsetAsync(loss, 0, sizeof(float), stream);

  dim3 block(256);
  dim3 grid(CEIL_DIV(b, block.x));

  cross_entropy_loss_kernel<<<grid, block, 0, stream>>>(A, target, loss, n, b);
  cudaStreamSynchronize(stream);

  CUDA_CHECK_KERNEL();
}

__global__ void cross_entropy_grad_kernel(const float *A, const float *target,
                                          float *dLdA, int n, int b) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < n && col < b) {
    int target_class = (int)target[col];
    float grad = A[cuda_utils::get_2d_idx(row, col, n)];

    if (row == target_class) {
      grad -= 1.0f; // Subtract 1 for the true class
    }

    dLdA[cuda_utils::get_2d_idx(row, col, n)] =
        grad / (float)b; // Average over batch
  }
}

void launch_cross_entropy_grad_kernel(const float *A, const float *target,
                                      float *dLdA, int n, int b,
                                      cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid(CEIL_DIV(n, block.x), CEIL_DIV(b, block.y));

  cross_entropy_grad_kernel<<<grid, block, 0, stream>>>(A, target, dLdA, n, b);
  cudaStreamSynchronize(stream);

  CUDA_CHECK_KERNEL();
}

__global__ void predict_class_kernel(const float *A, float *predictions, int n,
                                     int b) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < b) {
    float max_prob = A[cuda_utils::get_2d_idx(0, col, n)];
    int max_class = 0;

    for (int row = 1; row < n; ++row) {
      float prob = A[cuda_utils::get_2d_idx(row, col, n)];
      if (prob > max_prob) {
        max_prob = prob;
        max_class = row;
      }
    }

    predictions[col] = (float)max_class;
  }
}

void launch_predict_class_kernel(const float *A, float *predictions, int n,
                                 int b, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(CEIL_DIV(b, block.x));

  predict_class_kernel<<<grid, block, 0, stream>>>(A, predictions, n, b);
  cudaStreamSynchronize(stream);

  CUDA_CHECK_KERNEL();
}
