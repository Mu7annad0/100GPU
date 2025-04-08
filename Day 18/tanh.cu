#include <cuda_runtime.h>
#include <iostream>

__global__ void tanhKernel(const float* A, float* C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int idx = row * N + col;
        float exp_val = expf(2 * A[idx]);
        C[idx] = (exp_val - 1) / (exp_val + 1);
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    dim3 blockDim(32, 32);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 
                 (m + blockDim.y - 1) / blockDim.y);
    tanhKernel<<<gridDim, blockDim>>>(input, output, m, n);
    cudaDeviceSynchronize();
}