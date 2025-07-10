# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -O3
NVCCFLAGS = -std=c++17 -O3

# CUDA and library paths
CUDA_PATH = /usr/local/cuda
INCLUDES = -Iinclude -I$(CUDA_PATH)/include
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas

# Source files
CPP_SOURCES = src/tensor.cpp src/module.cpp src/mlp.cpp src/neural_network.cpp
CU_SOURCES = cuda/tensor_kernels.cu cuda/layer_kernels.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)

# Target
TARGET = main

# Default target
all: $(TARGET)

# Build main executable
$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS) main.o
	$(CXX) $^ -o $@ $(LIBS)

# Compile C++ files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Special rule for main.cpp
main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS) main.o $(TARGET)

# Help
help:
	@echo "Available targets:"
	@echo "  all    - Build the main executable"
	@echo "  run    - Build and run the program"
	@echo "  clean  - Remove all build files"
	@echo "  help   - Show this help message"

.PHONY: all run clean help
