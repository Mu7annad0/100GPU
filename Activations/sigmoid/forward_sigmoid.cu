#include <cuda_runtime.h>
#include "utils.cu"

__global__ void forward_sigmoid_fp32(float *inp, float *out, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N){
        out[tid] = 1.0f / (1.0f + expf(-inp[tid]));
    }
}

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
__global__ void forward_sigmoid_4fp32(float *inp, float *out, int N){
    int tid = (threadIdx.x + blockDim.x * blockIdx.x) * 4;

    if (tid < N){
        float4 in = FLOAT4(inp[tid]);
        float4 ot;

        ot.x = 1.0f / (1.0f + expf(-in.x));
        ot.y = 1.0f / (1.0f + expf(-in.y));
        ot.z = 1.0f / (1.0f + expf(-in.z));
        ot.w = 1.0f / (1.0f + expf(-in.w));

        FLOAT4(out[tid]) = ot;
    }
}

void run_forward_sigmoid(float *inp, float *out, int N){
    dim3 ThreadsPerBlock(1024);
    dim3 BlocksPerGrid((N + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    forward_sigmoid_4fp32<<<BlocksPerGrid, ThreadsPerBlock>>>(inp, out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
