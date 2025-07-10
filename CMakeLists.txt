cmake_minimum_required(VERSION 3.18)
project(MLP_CUDA LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(include)

# Add executable
add_executable(mlp_cuda
    main.cpp
    src/tensor.cpp
    src/module.cpp
    src/mlp.cpp
    cuda/tensor_kernels.cu
    cuda/layer_kernels.cu
)

# Link CUDA libraries
target_link_libraries(mlp_cuda CUDA::cudart CUDA::cublas)

# Set CUDA properties
set_property(TARGET mlp_cuda PROPERTY CUDA_STANDARD 17)
