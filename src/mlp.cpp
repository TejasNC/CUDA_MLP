#include "../include/mlp.hpp"

#include <cublas_v2.h>

extern cublasHandle_t global_cublas_handle;

MLP::MLP(int input_size, int output_size, int batch_size, const std::vector<int>& hidden_sizes,
         cudaStream_t stream)
    : input_size(input_size), output_size(output_size), batch_size(batch_size), stream(stream)
{

    // Build the network architecture
    int prev_size = input_size;

    // Add hidden layers with ReLU activation
    for (int hidden_size : hidden_sizes)
    {
        layers.push_back(std::make_unique<Linear>(prev_size, hidden_size, batch_size, stream));
        layers.push_back(std::make_unique<ReLU>(hidden_size, batch_size, stream));
        prev_size = hidden_size;
    }

    // Add output layer
    layers.push_back(std::make_unique<Linear>(prev_size, output_size, batch_size, stream));
    layers.push_back(std::make_unique<SoftMax>(output_size, batch_size, stream));

    // Create loss function
    loss_fn = std::make_unique<CrossEntropyLoss>(output_size, batch_size, stream);
}

const Tensor& MLP::forward(const Tensor& input)
{
    const Tensor* current_output = &input;

    for (auto& layer : layers)
    {
        current_output = &layer->forward(*current_output);
    }

    return *current_output;
}

float MLP::train_step(const Tensor& input, const Tensor& target, float learning_rate)
{
    // Forward pass
    const Tensor& predictions = forward(input);

    // Calculate loss
    float loss = loss_fn->calc_loss(predictions, target);

    // Calculate gradients from loss function
    const Tensor& loss_grad = loss_fn->calc_grad(predictions, target);

    // Backward pass through layers in reverse order
    const Tensor* grad_output = &loss_grad;
    for (int i = layers.size() - 1; i >= 0; --i)
    {
        grad_output = &layers[i]->backward(*grad_output);
    }

    // Update parameters
    update(learning_rate);

    return loss;
}

void MLP::update(float learning_rate)
{
    for (auto& layer : layers)
    {
        layer->update(learning_rate);
    }
}

float MLP::calc_loss(const Tensor& predictions, const Tensor& target)
{
    return loss_fn->calc_loss(predictions, target);
}

Tensor MLP::predict(const Tensor& input)
{
    const Tensor& predictions = forward(input);
    return loss_fn->predict_class(predictions);
}

float MLP::evaluate(const Tensor& input, const Tensor& target)
{
    const Tensor& predictions = forward(input);
    return calc_loss(predictions, target);
}
