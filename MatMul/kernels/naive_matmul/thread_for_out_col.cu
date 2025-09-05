#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixmulti1colforthread(float* A, float* B, float* C, int M, int K, int N){
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (col < N){
        for(int row = 0 ; row < M ; row++){
            float val = 0.0;
            for (int i = 0; i < K ; i++){
                val += A[row * K + i] * B[col + i * N];
            }
            C[col + row * N] = val;
        }
    }
}

void run_matmul_thread_one_col(float * A, float * B, float * C, int M, int K, int N){

    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    matrixmulti1colforthread<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}    
