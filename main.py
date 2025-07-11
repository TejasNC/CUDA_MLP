import torch
import torch.nn as nn
import time
import numpy as np
import statistics

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        layers = []

        # Input to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())

        # Final layer to output
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Network parameters
    input_size = 784    # 28x28 MNIST-like input
    output_size = 10    # 10 classes
    batch_size = 32
    hidden_sizes = [128, 64]  # Two hidden layers

    # Create MLP
    model = MLP(input_size, hidden_sizes, output_size).to(device)
    model.eval()  # Set to evaluation mode for inference

    print(f"Network Architecture: {input_size} -> {hidden_sizes[0]} -> {hidden_sizes[1]} -> {output_size}")
    print(f"Batch size: {batch_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create random input data (simulating MNIST-like inputs)
    torch.manual_seed(42)  # For reproducibility
    input_data = torch.randn(batch_size, input_size, device=device)

    print("\nStarting inference timing...")

    # Warm-up runs
    num_warmup = 10
    print(f"Performing {num_warmup} warm-up runs...")

    with torch.no_grad():
        for i in range(num_warmup):
            _ = model(input_data)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timed inference runs
    num_runs = 100
    inference_times = []

    print(f"Running {num_runs} timed inference iterations...")

    with torch.no_grad():
        for i in range(num_runs):
            # Start timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Run inference
            predictions = model(input_data)

            # Ensure GPU operations are complete
            if device.type == "cuda":
                torch.cuda.synchronize()

            # End timing
            end_time = time.perf_counter()

            # Calculate duration in microseconds
            duration_us = (end_time - start_time) * 1_000_000
            inference_times.append(duration_us)

            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1} runs...")

    # Calculate statistics
    avg_time = statistics.mean(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    std_dev = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0

    # Print results
    print("\n=== INFERENCE TIMING RESULTS ===")
    print(f"Number of runs: {num_runs}")
    print(f"Average time per inference: {avg_time:.2f} μs ({avg_time / 1000:.3f} ms)")
    print(f"Minimum time: {min_time:.2f} μs ({min_time / 1000:.3f} ms)")
    print(f"Maximum time: {max_time:.2f} μs ({max_time / 1000:.3f} ms)")
    print(f"Standard deviation: {std_dev:.2f} μs")
    print(f"Throughput: {(batch_size * 1_000_000) / avg_time:.0f} samples/second")

    # Show a sample prediction result
    with torch.no_grad():
        final_predictions = model(input_data)
        predicted_classes = torch.argmax(final_predictions, dim=1)

    print("\nSample predictions (first 5 samples):")
    for i in range(min(5, batch_size)):
        print(f"Sample {i}: Predicted class = {predicted_classes[i].item()}")

    print("\nInference timing completed successfully!")

if __name__ == "__main__":
    main()
