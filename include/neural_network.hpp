#pragma once

#include <memory>
#include <string>
#include <vector>

// Forward declarations to hide CUDA details
class MLP;
class Tensor;
typedef struct CUstream_st*   cudaStream_t;
typedef struct cublasContext* cublasHandle_t;

/**
 * High-level Neural Network interface that hides all CUDA and tensor complexity
 * Usage:
 *   NeuralNetwork nn({784, 128, 64, 10});
 *   nn.load_data("train.csv", "test.csv");
 *   nn.train(epochs=10, learning_rate=0.001);
 *   float accuracy = nn.evaluate();
 */
class NeuralNetwork
{
  public:
    // Constructor: specify layer sizes
    NeuralNetwork(const std::vector<int>& layer_sizes, int batch_size = 100);

    // Destructor
    ~NeuralNetwork();

    // Data loading
    void load_data(const std::string& train_csv_path, const std::string& test_csv_path = "");
    void load_training_data(const std::vector<std::vector<float>>& images,
                            const std::vector<int>&                labels);
    void load_test_data(const std::vector<std::vector<float>>& images,
                        const std::vector<int>&                labels);

    // Training
    void  train(int num_epochs = 10, float learning_rate = 0.001f, bool verbose = true);
    float train_epoch(float learning_rate = 0.001f);

    // Evaluation
    float            evaluate();      // Returns test accuracy
    float            evaluate_loss(); // Returns test loss
    std::vector<int> predict(const std::vector<std::vector<float>>& images);
    int              predict_single(const std::vector<float>& image);

    // Model persistence
    void save_model(const std::string& filepath);
    void load_model(const std::string& filepath);

    // Utilities
    void               set_batch_size(int batch_size);
    void               print_architecture();
    std::vector<float> get_loss_history();
    std::vector<float> get_accuracy_history();

    // Benchmarking
    double get_training_time();  // Total training time in seconds
    double get_inference_time(); // Average inference time per sample
    void   benchmark_inference(int num_samples = 1000);

  private:
    // Implementation details (hidden from user)
    struct Impl;
    std::unique_ptr<Impl> pImpl;

    // No copy/assignment (move-only)
    NeuralNetwork(const NeuralNetwork&)            = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&)                 = default;
    NeuralNetwork& operator=(NeuralNetwork&&)      = default;
};

// Convenience factory functions
NeuralNetwork create_mnist_network(int batch_size = 100);
NeuralNetwork create_custom_network(const std::vector<int>& layers, int batch_size = 100);
