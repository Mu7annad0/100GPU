#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for vector Addition
__global__ void vecAddKernel(float* A, float* B, float* C, float n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}

int main(){
    const int n = 10;
    int size = n* sizeof(float);
    float A_h[n], B_h[n], C_h[n];
    float *A_d, *B_d, *C_d;

    // Allocate device memory for A, B and C 
    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);
    cudaMalloc(&C_d, size);

    // Copy A and B to device memory
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Call kernel to launch a grid of threads to perform the actual vector addidtion
    int blockDim = 256;
    int gridDim = ceil(n/blockDim);
    vecAddKernel<<<gridDim,blockDim>>>(A_d, B_d, C_d, n);

    // Copy C from the device memory
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Free device vectors
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
