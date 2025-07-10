#include "include/mlp.hpp"

#include <iostream>
#include <random>
#include <vector>

// Global cuBLAS handle
cublasHandle_t global_cublas_handle;

int main()
{
    // Initialize CUDA and cuBLAS
    CUDA_CHECK(cudaSetDevice(0));

    cublasStatus_t status = cublasCreate(&global_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create cuBLAS handle" << std::endl;
        return -1;
    }

    try
    {
        // Network parameters
        int              input_size   = 784; // 28x28 MNIST-like input
        int              output_size  = 10;  // 10 classes
        int              batch_size   = 32;
        std::vector<int> hidden_sizes = {128, 64}; // Two hidden layers

        // Create CUDA stream
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Create MLP
        MLP mlp(input_size, output_size, batch_size, hidden_sizes, stream);

        // Create dummy data for testing
        Tensor input(input_size, batch_size, stream);
        Tensor target(1, batch_size, stream);

        // Initialize with random data
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int>    target_dist(0, output_size - 1);

        std::vector<float> h_input(input_size * batch_size);
        std::vector<float> h_target(batch_size);

        for (int i = 0; i < input_size * batch_size; ++i)
        {
            h_input[i] = input_dist(gen);
        }

        for (int i = 0; i < batch_size; ++i)
        {
            h_target[i] = static_cast<float>(target_dist(gen));
        }

        input.copy_from_host(h_input);
        target.copy_from_host(h_target);

        // Training parameters
        float learning_rate = 0.001f;
        int   num_epochs    = 10;

        std::cout << "Starting training..." << std::endl;

        // Training loop
        for (int epoch = 0; epoch < num_epochs; ++epoch)
        {
            float loss = mlp.train_step(input, target, learning_rate);
            std::cout << "Epoch " << epoch + 1 << ", Loss: " << loss << std::endl;
        }

        // Ensure all training operations are complete before prediction
        CUDA_CHECK(cudaDeviceSynchronize());

        // Test prediction
        Tensor             predictions   = mlp.predict(input);
        std::vector<float> h_predictions = predictions.to_cpu();

        std::cout << "\nPredictions for first 5 samples:" << std::endl;
        for (int i = 0; i < std::min(5, batch_size); ++i)
        {
            std::cout << "Sample " << i
                      << ": Predicted class = " << static_cast<int>(h_predictions[i])
                      << ", True class = " << static_cast<int>(h_target[i]) << std::endl;
        }

        // Cleanup
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        cublasDestroy(global_cublas_handle);
        return -1;
    }

    // Cleanup cuBLAS
    cublasDestroy(global_cublas_handle);

    std::cout << "\nTraining completed successfully!" << std::endl;
    return 0;
}
