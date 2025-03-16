#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for matrix addition
__global__ void matrixAddKernel(float* A, float* B, float* C, int N){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if ((col >= N) || (row >= N)) { return ; }

    C[col+row*N] = A[col+row*N] + B[col+row*N];
}

int main(){
    const int n = 5;
    int size = n*n* sizeof(float);
    float *A_d, *B_d, *C_d;

    float *A_h = (float*)malloc(size);
    float *B_h = (float*)malloc(size);
    float *C_h = (float*)malloc(size);

    for (int i=0; i < n; i++){
        for (int j=0; j < n; j++){
            A_h[i * n + j] = (i == j) ? 1.0f : 0.0f;
            B_h[i * n + j] = i*n - j + 3;
            C_h[i * n + j] = 0.0f;
        }
    }
    // Print the input matrices
    printf("Matrix A (Identity):\n");
    for (int i=0; i < n; i++){
        for (int j=0; j < n; j++){
            printf("%.1f ", A_h[i * n + j]);
        }
        printf("\n");
    }
    
    printf("\nMatrix B:\n");
    for (int i=0; i < n; i++){
        for (int j=0; j < n; j++){
            printf("%.1f ", B_h[i * n + j]);
        }
        printf("\n");
    }

    // Allocate device memory for A, B and C matrices
    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);
    cudaMalloc(&C_d, size);

    // Copy A and B matrices to device memory
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Call kernel to launch a grid of threads to perform the matrix addition
    dim3 blockDim(32, 16);
    dim3 gridDim((int)ceil(n / 32.0f), (int)ceil(n/ 16.0f));
    matrixAddKernel<<<gridDim,blockDim>>>(A_d, B_d, C_d, n);
    cudaDeviceSynchronize();

    // Copy result matrix C from the device memory to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    printf("\nOutput:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            printf("%.2f ",C_h[i * n + j]); // Prints each element with 2 decimal precision
        }
        printf("\n"); // Adds a newline after each row
    }
    // Free device matrices
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
    // Free host matrices
    free(A_h);
    free(B_h);
    free(C_h);
    
    return 0;
}
