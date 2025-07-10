#include "../include/neural_network.hpp"

#include "../include/cuda_utils.hpp"
#include "../include/mlp.hpp"
#include "../include/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

// Global cuBLAS handle (managed internally)
extern cublasHandle_t global_cublas_handle;

// PIMPL implementation to hide CUDA details
struct NeuralNetwork::Impl
{
    std::unique_ptr<MLP> mlp;
    cudaStream_t         stream;
    int                  batch_size;

    // Data storage
    std::vector<float> train_images;
    std::vector<float> train_labels;
    std::vector<float> test_images;
    std::vector<float> test_labels;
    int                num_train_samples = 0;
    int                num_test_samples  = 0;
    int                input_size        = 784;
    int                output_size       = 10;

    // Training history
    std::vector<float> loss_history;
    std::vector<float> accuracy_history;

    // Timing
    std::chrono::duration<double> total_training_time{0};
    std::chrono::duration<double> total_inference_time{0};
    int                           inference_count = 0;

    Impl(const std::vector<int>& layer_sizes, int batch_size) : batch_size(batch_size)
    {

        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Extract network dimensions
        input_size  = layer_sizes[0];
        output_size = layer_sizes.back();

        // Create hidden layer sizes
        std::vector<int> hidden_sizes;
        for (int i = 1; i < layer_sizes.size() - 1; ++i)
        {
            hidden_sizes.push_back(layer_sizes[i]);
        }

        // Create MLP
        mlp = std::make_unique<MLP>(input_size, output_size, batch_size, hidden_sizes, stream);
    }

    ~Impl()
    {
        if (stream)
        {
            cudaStreamDestroy(stream);
        }
    }

    void load_csv(const std::string& filepath, std::vector<float>& images,
                  std::vector<float>& labels, int& num_samples)
    {

        std::ifstream file(filepath);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        std::string line;
        std::getline(file, line); // Skip header

        std::vector<std::vector<float>> temp_images;
        std::vector<int>                temp_labels;

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string       cell;

            // First column: label
            std::getline(ss, cell, ',');
            int label = std::stoi(cell);
            temp_labels.push_back(label);

            // Next 784 columns: pixels
            std::vector<float> pixels(input_size);
            for (int i = 0; i < input_size; ++i)
            {
                std::getline(ss, cell, ',');
                pixels[i] = std::stof(cell) / 255.0f; // Normalize to [0,1]
            }
            temp_images.push_back(pixels);
        }

        num_samples = temp_images.size();

        // Flatten images for efficient access
        images.resize(num_samples * input_size);
        labels.resize(num_samples);

        for (int i = 0; i < num_samples; ++i)
        {
            labels[i] = static_cast<float>(temp_labels[i]);
            for (int j = 0; j < input_size; ++j)
            {
                images[i * input_size + j] = temp_images[i][j];
            }
        }

        std::cout << "Loaded " << num_samples << " samples from " << filepath << std::endl;
    }

    void shuffle_training_data()
    {
        std::vector<int> indices(num_train_samples);
        std::iota(indices.begin(), indices.end(), 0);

        std::random_device rd;
        std::mt19937       g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        std::vector<float> shuffled_images(train_images.size());
        std::vector<float> shuffled_labels(train_labels.size());

        for (int i = 0; i < num_train_samples; ++i)
        {
            shuffled_labels[i] = train_labels[indices[i]];
            for (int j = 0; j < input_size; ++j)
            {
                shuffled_images[i * input_size + j] = train_images[indices[i] * input_size + j];
            }
        }

        train_images = std::move(shuffled_images);
        train_labels = std::move(shuffled_labels);
    }

    void get_batch(const std::vector<float>& images, const std::vector<float>& labels,
                   int batch_idx, int actual_batch_size, std::vector<float>& batch_images,
                   std::vector<float>& batch_labels)
    {

        batch_images.resize(actual_batch_size * input_size);
        batch_labels.resize(actual_batch_size);

        int start_idx = batch_idx * batch_size;
        for (int i = 0; i < actual_batch_size; ++i)
        {
            batch_labels[i] = labels[start_idx + i];
            for (int j = 0; j < input_size; ++j)
            {
                batch_images[i * input_size + j] = images[(start_idx + i) * input_size + j];
            }
        }
    }
};

// Constructor
NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes, int batch_size)
    : pImpl(std::make_unique<Impl>(layer_sizes, batch_size))
{
}

// Destructor
NeuralNetwork::~NeuralNetwork() = default;

// Data loading
void NeuralNetwork::load_data(const std::string& train_csv_path, const std::string& test_csv_path)
{
    pImpl->load_csv(train_csv_path, pImpl->train_images, pImpl->train_labels,
                    pImpl->num_train_samples);

    if (!test_csv_path.empty())
    {
        pImpl->load_csv(test_csv_path, pImpl->test_images, pImpl->test_labels,
                        pImpl->num_test_samples);
    }
}

void NeuralNetwork::load_training_data(const std::vector<std::vector<float>>& images,
                                       const std::vector<int>&                labels)
{
    pImpl->num_train_samples = images.size();
    pImpl->train_images.resize(pImpl->num_train_samples * pImpl->input_size);
    pImpl->train_labels.resize(pImpl->num_train_samples);

    for (int i = 0; i < pImpl->num_train_samples; ++i)
    {
        pImpl->train_labels[i] = static_cast<float>(labels[i]);
        for (int j = 0; j < pImpl->input_size; ++j)
        {
            pImpl->train_images[i * pImpl->input_size + j] = images[i][j];
        }
    }
}

void NeuralNetwork::load_test_data(const std::vector<std::vector<float>>& images,
                                   const std::vector<int>&                labels)
{
    pImpl->num_test_samples = images.size();
    pImpl->test_images.resize(pImpl->num_test_samples * pImpl->input_size);
    pImpl->test_labels.resize(pImpl->num_test_samples);

    for (int i = 0; i < pImpl->num_test_samples; ++i)
    {
        pImpl->test_labels[i] = static_cast<float>(labels[i]);
        for (int j = 0; j < pImpl->input_size; ++j)
        {
            pImpl->test_images[i * pImpl->input_size + j] = images[i][j];
        }
    }
}

// Training
void NeuralNetwork::train(int num_epochs, float learning_rate, bool verbose)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    if (verbose)
    {
        std::cout << "Starting training for " << num_epochs << " epochs..." << std::endl;
        std::cout << "Batch size: " << pImpl->batch_size << std::endl;
        std::cout << "Learning rate: " << learning_rate << std::endl;
    }

    pImpl->loss_history.clear();
    pImpl->accuracy_history.clear();

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        float epoch_loss = train_epoch(learning_rate);
        pImpl->loss_history.push_back(epoch_loss);

        if (verbose)
        {
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << ", Loss: " << epoch_loss;

            // Calculate accuracy if test data available
            if (pImpl->num_test_samples > 0)
            {
                float accuracy = evaluate();
                pImpl->accuracy_history.push_back(accuracy);
                std::cout << ", Test Accuracy: " << (accuracy * 100.0f) << "%";
            }
            std::cout << std::endl;
        }
    }

    auto end_time              = std::chrono::high_resolution_clock::now();
    pImpl->total_training_time = end_time - start_time;

    if (verbose)
    {
        std::cout << "Training completed in " << pImpl->total_training_time.count() << " seconds"
                  << std::endl;
    }
}

float NeuralNetwork::train_epoch(float learning_rate)
{
    // Shuffle training data
    pImpl->shuffle_training_data();

    int   num_batches = pImpl->num_train_samples / pImpl->batch_size;
    float total_loss  = 0.0f;

    // Create reusable tensors for this epoch
    Tensor batch_input(pImpl->input_size, pImpl->batch_size, pImpl->stream);
    Tensor batch_target(1, pImpl->batch_size, pImpl->stream);

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        std::vector<float> h_batch_images, h_batch_labels;
        pImpl->get_batch(pImpl->train_images, pImpl->train_labels, batch_idx, pImpl->batch_size,
                         h_batch_images, h_batch_labels);

        // Copy to GPU
        batch_input.copy_from_host(h_batch_images);
        batch_target.copy_from_host(h_batch_labels);

        // Train step
        float batch_loss = pImpl->mlp->train_step(batch_input, batch_target, learning_rate);
        total_loss += batch_loss;
    }

    return total_loss / num_batches;
}

// Evaluation
float NeuralNetwork::evaluate()
{
    if (pImpl->num_test_samples == 0)
    {
        throw std::runtime_error("No test data loaded");
    }

    int num_batches         = pImpl->num_test_samples / pImpl->batch_size;
    int correct_predictions = 0;
    int total_predictions   = 0;

    Tensor batch_input(pImpl->input_size, pImpl->batch_size, pImpl->stream);

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        std::vector<float> h_batch_images, h_batch_labels;
        pImpl->get_batch(pImpl->test_images, pImpl->test_labels, batch_idx, pImpl->batch_size,
                         h_batch_images, h_batch_labels);

        batch_input.copy_from_host(h_batch_images);

        Tensor             predictions   = pImpl->mlp->predict(batch_input);
        std::vector<float> h_predictions = predictions.to_cpu();

        // Count correct predictions
        for (int i = 0; i < pImpl->batch_size; ++i)
        {
            if (static_cast<int>(h_predictions[i]) == static_cast<int>(h_batch_labels[i]))
            {
                correct_predictions++;
            }
            total_predictions++;
        }
    }

    return static_cast<float>(correct_predictions) / total_predictions;
}

float NeuralNetwork::evaluate_loss()
{
    if (pImpl->num_test_samples == 0)
    {
        throw std::runtime_error("No test data loaded");
    }

    int   num_batches = pImpl->num_test_samples / pImpl->batch_size;
    float total_loss  = 0.0f;

    Tensor batch_input(pImpl->input_size, pImpl->batch_size, pImpl->stream);
    Tensor batch_target(1, pImpl->batch_size, pImpl->stream);

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        std::vector<float> h_batch_images, h_batch_labels;
        pImpl->get_batch(pImpl->test_images, pImpl->test_labels, batch_idx, pImpl->batch_size,
                         h_batch_images, h_batch_labels);

        batch_input.copy_from_host(h_batch_images);
        batch_target.copy_from_host(h_batch_labels);

        float batch_loss = pImpl->mlp->evaluate(batch_input, batch_target);
        total_loss += batch_loss;
    }

    return total_loss / num_batches;
}

std::vector<int> NeuralNetwork::predict(const std::vector<std::vector<float>>& images)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<int> predictions;
    int              num_samples = images.size();
    int              num_batches = (num_samples + pImpl->batch_size - 1) / pImpl->batch_size;

    Tensor batch_input(pImpl->input_size, pImpl->batch_size, pImpl->stream);

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        int start_idx         = batch_idx * pImpl->batch_size;
        int end_idx           = std::min(start_idx + pImpl->batch_size, num_samples);
        int actual_batch_size = end_idx - start_idx;

        // Prepare batch
        std::vector<float> h_batch_images(actual_batch_size * pImpl->input_size);
        for (int i = 0; i < actual_batch_size; ++i)
        {
            for (int j = 0; j < pImpl->input_size; ++j)
            {
                h_batch_images[i * pImpl->input_size + j] = images[start_idx + i][j];
            }
        }

        // If partial batch, pad with zeros
        if (actual_batch_size < pImpl->batch_size)
        {
            h_batch_images.resize(pImpl->batch_size * pImpl->input_size, 0.0f);
        }

        batch_input.copy_from_host(h_batch_images);
        Tensor             batch_predictions = pImpl->mlp->predict(batch_input);
        std::vector<float> h_predictions     = batch_predictions.to_cpu();

        // Extract only the valid predictions
        for (int i = 0; i < actual_batch_size; ++i)
        {
            predictions.push_back(static_cast<int>(h_predictions[i]));
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    pImpl->total_inference_time += (end_time - start_time);
    pImpl->inference_count += num_samples;

    return predictions;
}

int NeuralNetwork::predict_single(const std::vector<float>& image)
{
    std::vector<std::vector<float>> single_image = {image};
    std::vector<int>                predictions  = predict(single_image);
    return predictions[0];
}

// Utilities
void NeuralNetwork::set_batch_size(int batch_size)
{
    pImpl->batch_size = batch_size;
    // Note: Would need to recreate MLP for this to take effect
    std::cout << "Warning: Batch size change requires network recreation" << std::endl;
}

void NeuralNetwork::print_architecture()
{
    std::cout << "Neural Network Architecture:" << std::endl;
    std::cout << "Input size: " << pImpl->input_size << std::endl;
    std::cout << "Output size: " << pImpl->output_size << std::endl;
    std::cout << "Batch size: " << pImpl->batch_size << std::endl;
    std::cout << "Training samples: " << pImpl->num_train_samples << std::endl;
    std::cout << "Test samples: " << pImpl->num_test_samples << std::endl;
}

std::vector<float> NeuralNetwork::get_loss_history() { return pImpl->loss_history; }

std::vector<float> NeuralNetwork::get_accuracy_history() { return pImpl->accuracy_history; }

double NeuralNetwork::get_training_time() { return pImpl->total_training_time.count(); }

double NeuralNetwork::get_inference_time()
{
    if (pImpl->inference_count == 0)
        return 0.0;
    return pImpl->total_inference_time.count() / pImpl->inference_count;
}

void NeuralNetwork::benchmark_inference(int num_samples)
{
    // Create random test data
    std::vector<std::vector<float>> test_images(num_samples,
                                                std::vector<float>(pImpl->input_size, 0.5f));

    auto start_time = std::chrono::high_resolution_clock::now();
    predict(test_images);
    auto end_time = std::chrono::high_resolution_clock::now();

    double time_taken = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Inference benchmark: " << num_samples << " samples in " << time_taken
              << " seconds" << std::endl;
    std::cout << "Throughput: " << (num_samples / time_taken) << " samples/second" << std::endl;
    std::cout << "Latency: " << (time_taken / num_samples * 1000.0) << " ms/sample" << std::endl;
}

// Factory functions
NeuralNetwork create_mnist_network(int batch_size)
{
    return NeuralNetwork({784, 128, 64, 10}, batch_size);
}

NeuralNetwork create_custom_network(const std::vector<int>& layers, int batch_size)
{
    return NeuralNetwork(layers, batch_size);
}
