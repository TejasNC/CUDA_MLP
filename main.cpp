#include "include/mlp.hpp"

#include <chrono>
#include <cmath>
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

        // Create input tensor for inference
        Tensor input(input_size, batch_size, stream);

        // Initialize with random data (simulating MNIST-like inputs)
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);

        std::vector<float> h_input(input_size * batch_size);
        for (int i = 0; i < input_size * batch_size; ++i)
        {
            h_input[i] = input_dist(gen);
        }
        input.copy_from_host(h_input);

        std::cout << "Network Architecture: " << input_size << " -> " << hidden_sizes[0] << " -> "
                  << hidden_sizes[1] << " -> " << output_size << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "\nStarting inference timing..." << std::endl;

        // Warm-up runs
        const int num_warmup = 10;
        std::cout << "Performing " << num_warmup << " warm-up runs..." << std::endl;
        for (int i = 0; i < num_warmup; ++i)
        {
            Tensor predictions = mlp.predict(input);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Timed inference runs
        const int           num_runs = 100;
        std::vector<double> inference_times;
        inference_times.reserve(num_runs);

        std::cout << "Running " << num_runs << " timed inference iterations..." << std::endl;

        for (int i = 0; i < num_runs; ++i)
        {
            // Start timing
            auto start = std::chrono::high_resolution_clock::now();

            // Run inference
            Tensor predictions = mlp.predict(input);

            // Ensure GPU operations are complete
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // End timing
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate duration in microseconds
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            inference_times.push_back(duration.count());

            if ((i + 1) % 20 == 0)
            {
                std::cout << "Completed " << (i + 1) << " runs..." << std::endl;
            }
        }

        // Calculate statistics
        double total_time = 0.0;
        double min_time   = inference_times[0];
        double max_time   = inference_times[0];

        for (double time : inference_times)
        {
            total_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }

        double avg_time = total_time / num_runs;

        // Calculate standard deviation
        double variance = 0.0;
        for (double time : inference_times)
        {
            variance += (time - avg_time) * (time - avg_time);
        }
        variance /= num_runs;
        double std_dev = std::sqrt(variance);

        // Print results
        std::cout << "\n=== INFERENCE TIMING RESULTS ===" << std::endl;
        std::cout << "Number of runs: " << num_runs << std::endl;
        std::cout << "Average time per inference: " << avg_time << " μs (" << avg_time / 1000.0
                  << " ms)" << std::endl;
        std::cout << "Minimum time: " << min_time << " μs (" << min_time / 1000.0 << " ms)"
                  << std::endl;
        std::cout << "Maximum time: " << max_time << " μs (" << max_time / 1000.0 << " ms)"
                  << std::endl;
        std::cout << "Standard deviation: " << std_dev << " μs" << std::endl;
        std::cout << "Throughput: " << (batch_size * 1000000.0) / avg_time << " samples/second"
                  << std::endl;

        // Show a sample prediction result
        Tensor             final_predictions = mlp.predict(input);
        std::vector<float> h_predictions     = final_predictions.to_cpu();

        std::cout << "\nSample predictions (first 5 samples):" << std::endl;
        for (int i = 0; i < std::min(5, batch_size); ++i)
        {
            std::cout << "Sample " << i
                      << ": Predicted class = " << static_cast<int>(h_predictions[i]) << std::endl;
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

    std::cout << "\nInference timing completed successfully!" << std::endl;
    return 0;
}
