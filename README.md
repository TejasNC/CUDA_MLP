# CUDA MLP - Custom Neural Network Implementation

A from-scratch CUDA implementation of Multi-Layer Perceptron for MNIST classification. Built to understand GPU computing and neural network fundamentals at a low level.

## Current Status

**Working:**
- CUDA memory management (tensors, streams, device allocation)
- Forward/backward pass kernels (linear layers, ReLU, softmax, cross-entropy)
- High-level C++ API that hides CUDA complexity
- Comprehensive benchmarking suite vs NumPy/PyTorch
- Clean build system (CMake + Makefile)

**Issues:**
- Training accuracy stuck at ~10% (random guessing level) for MNIST
- Likely bugs in gradient computation or weight updates
- Need debugging of numerical implementation
