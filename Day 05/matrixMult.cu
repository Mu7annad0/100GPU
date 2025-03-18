#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for Matrix multiplication
// A (M x N), B (N x K), C (M x K)
__global__ void matrixMultKernel(float* A, float* B, float* C, int M, int N, int K) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < K && row < M) {
        float val = 0.0f;
        for (int i = 0; i < N; i++) {
            val += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = val;
    }
}

int main() {
    // Define matrix dimensions
    int M = 2;
    int N = 4;
    int K = 3;

    int size_A = M * N * sizeof(float);
    int size_B = N * K * sizeof(float);
    int size_C = M * K * sizeof(float);

    // Allocate host memory
    float *A_h = (float*)malloc(size_A);
    float *B_h = (float*)malloc(size_B);
    float *C_h = (float*)malloc(size_C);
    
    // Initialize input with sample data for matrix A (M x N)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A_h[i * N + j] = 1.0f + i * N + j;
        }
    }
    
    // Initialize input with sample data for matrix B (N x K)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            B_h[i * K + j] = 2.0f + i * K + j;
        }
    }

    // Print matrix A
    printf("Matrix A (M=%d x N=%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", A_h[i * N + j]);
        }
        printf("\n");
    }

    // Print matrix B
    printf("\nMatrix B (N=%d x K=%d):\n", N, K);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            printf("%.2f ", B_h[i * K + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    // Allocate device memory
    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, size_A);
    cudaMalloc(&B_d, size_B);
    cudaMalloc(&C_d, size_C);

    // Copy input data to device
    cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((int)ceil((float)K/16), (int)ceil((float)M/16));
    
    // Launch kernel
    matrixMultKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, N, K);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost);

    // Print results
    printf("Result matrix C (M=%d x K=%d):\n", M, K);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%.2f ", C_h[i * K + j]);
        }
        printf("\n");
    }

    // Free memory
    free(A_h);
    free(B_h);
    free(C_h);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
    return 0;
}