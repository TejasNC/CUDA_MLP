# CUDA MLP - Custom Neural Network Implementation

A from-scratch CUDA implementation of Multi-Layer Perceptron for MNIST classification. Built to understand GPU computing and neural network fundamentals at a low level.

## Current Status

**Working:**
- CUDA memory management (tensors, streams, device allocation)
- Forward/backward pass kernels (linear layers, ReLU, softmax, cross-entropy)
- High-level C++ API that hides CUDA complexity
- Comprehensive benchmarking suite vs NumPy/PyTorch
- Clean build system (CMake + Makefile)

## Performance Benchmarks

https://www.kaggle.com/code/tejaszz/benchmarking

**Inference Performance (784→128→64→10 architecture, batch size 32):**

| Implementation | Avg Time | Min Time | Max Time | Std Dev | Throughput |
|---|---|---|---|---|---|
| **Custom CUDA** | **120.36 μs** | **115 μs** | **148 μs** | **6.28 μs** | **265,869 samples/s** |
| PyTorch | 156.88 μs | 135.81 μs | 340.10 μs | 38.63 μs | 203,974 samples/s |

**Key Results:**
- **30% faster** than PyTorch for inference
- **6x better timing consistency** (lower standard deviation)
- **61,895 more samples/second** throughput
- Benchmarking framework includes warm-up runs and statistical analysis across 100 iterations

**Issues:**
- Training accuracy stuck at ~10% (random guessing level) for MNIST
- Likely bugs in gradient computation or weight updates
- Need debugging of numerical implementation

**Resources:**

I made Notion notes while learning CUDA Programming which can be found here:
https://www.notion.so/CUDA-Programming-227209f8df35808fae45f80159cce0ab?source=copy_link
