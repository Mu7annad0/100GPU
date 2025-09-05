#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrix_multi(float* A, float* B, float* out, int M, int K, int N){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < M && col < N){
        float val = 0.0;
        for (int i = 0 ; i < K ; i++){
            val += A[row * K + i] * B[i * N + col];
        }
        out[row*N+col] = val;
    }
}

void run_matmul_thread_one_col(float * A, float * B, float * C, int M, int K, int N){

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    matrix_multi<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}    
