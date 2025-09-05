#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixmulti1rowforthread(float* A, float* B, float* C, int M, int K, int N){
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < M){
        for(int col =0 ; col < N ; col++){
            float val = 0.0;
            for (int i = 0; i < K ; i++){
                val += A[i + row * K] * B[col + i * N];
            }
            C[col + row * N] = val;
        }
    }
}

void run_matmul_thread_one_row(float * A, float * B, float * C, int M, int K, int N){

    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    matrixmulti1rowforthread<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}    