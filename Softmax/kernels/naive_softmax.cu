#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.cu"

__global__ void naive_softmax_kernel(float *inp, float *out, int M, int N){

    int row = threadIdx.x + blockDim.x * blockIdx.x;

    // STEP 1 -->> CALCULATE THE MAXIMUM
    float max_val = -1 * INFINITY;
    for (int col = 0; col < N; col++){
        int i = row * N + col;
        max_val = fmax(max_val, inp[i]);
    }

    // STEP 2 -->> CALCULATE THE NORMALIZATION
    float norm_val = 0.0f;
    for (int col = 0; col < N; col++){
        int i = row * N + col;
        norm_val += expf(inp[i] - max_val);
    }

    // STEP 3 -->> CALCULATE THE SOFTMAX
    for (int col = 0; col < N; col++){
        int i = row * N + col;
        out[i] = expf(inp[i] - max_val) / norm_val;
    }
}

void run_naive_softmax(float *inp, float *out, int M, int N){
    dim3 ThreadsPerBlock(1024);
    dim3 BlocksPerGrid((M + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    naive_softmax_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(inp, out, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}