#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int         deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    std::cout << "CUDA headers found successfully!" << std::endl;
    std::cout << "cudaGetDeviceCount returned: " << cudaGetErrorString(error) << std::endl;

    return 0;
}
