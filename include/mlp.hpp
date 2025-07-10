#pragma once

#include "../include/cuda_utils.hpp"
#include "../include/module.hpp"
#include "../include/tensor.hpp"

#include <cublas_v2.h>
#include <memory>
#include <vector>

class MLP
{
  private:
    std::vector<std::unique_ptr<Module>> layers;
    std::unique_ptr<CrossEntropyLoss>    loss_fn;
    cudaStream_t                         stream;
    int                                  input_size;
    int                                  output_size;
    int                                  batch_size;

  public:
    MLP(int input_size, int output_size, int batch_size, const std::vector<int>& hidden_sizes,
        cudaStream_t stream = 0);

    ~MLP() = default;

    // Forward pass
    const Tensor& forward(const Tensor& input);

    // Update parameters
    void update(float learning_rate);

    // Calculate loss
    float calc_loss(const Tensor& predictions, const Tensor& target);

    // Make predictions
    Tensor predict(const Tensor& input);

    // Training step (combines forward, backward, and update)
    float train_step(const Tensor& input, const Tensor& target, float learning_rate);

    // Evaluation
    float evaluate(const Tensor& input, const Tensor& target);
};
