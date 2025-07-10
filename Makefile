# Makefile for MLP-CUDA project

# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -O3
NVCCFLAGS = -std=c++17 -O3

# CUDA paths (adjust if different)
CUDA_PATH = /usr/local/cuda
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

# Include paths
INCLUDES = -Iinclude -I$(CUDA_INCLUDE)

# Library paths
LIBS = -L$(CUDA_LIB) -lcudart

# Source files
CPP_SOURCES = src/tensor.cpp src/layer.cpp src/mlp.cpp
CU_SOURCES = cuda/tensor_kernels.cu cuda/layer_kernels.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

# Target executable
TARGET = mlp_cuda

# Build rules
all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CXX) $(CPP_OBJECTS) $(CU_OBJECTS) -o $@ $(LIBS)

# Compile C++ files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS) $(TARGET)

# Phony targets
.PHONY: all clean
