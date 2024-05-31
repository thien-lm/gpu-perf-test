#include <iostream>
#include <cmath>

// CUDA kernel for vector addition
__global__ void vectorAdd(float* A, float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int n = 4096*4096; // Number of elements in the arrays
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    // Allocate memory on the host
    float* h_A = new float[n];
    float* h_B = new float[n];
    float* h_C = new float[n];

    // Initialize input arrays
    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate memory on the device (GPU)
    float* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_C, n * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first few elements of the result
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
