#include <cuda_runtime.h>
#include "utils.cu"

__global__ void backward_sigmoid_fp32(float *out, float *din, float *dout, int n){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < n){
        din[tid] = dout[tid] * out[tid] * (1.0f - out[tid]);
    }
}

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
__global__ void backward_sigmoid_4fp32(float *out, float *din, float *dout, int n){
    int tid = (threadIdx.x + blockDim.x * blockIdx.x) * 4;
    if (tid + 3 < n){
        float4 ot = FLOAT4(out[tid]);
        float4 dt = FLOAT4(dout[tid]);
        float4 dx;

        dx.x = dt.x * ot.x * (1.0f - ot.x);
        dx.y = dt.y * ot.y * (1.0f - ot.y);
        dx.z = dt.z * ot.z * (1.0f - ot.z);
        dx.w = dt.w * ot.w * (1.0f - ot.w);

        FLOAT4(din[tid]) = dx;
    }
}

void run_backward_sigmoid(float *out, float *din, float *dout, int n){
    dim3 ThreadsPerBlock(1024);
    dim3 BlocksPerGrid((n + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    backward_sigmoid_4fp32<<<BlocksPerGrid, ThreadsPerBlock>>>(out, din, dout, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}