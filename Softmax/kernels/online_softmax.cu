#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void online_softmax_kernel(float* inp, float* out, int M, int N){
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < M){
        float max_val = -1 * INFINITY;
        float norm_val = 0.0f;
        for (int col = 0; col < N; col++){
            float curr = inp[row*N + col];
            if (curr > max_val){
                norm_val = norm_val * expf(max_val - curr);
                max_val = curr;
            }
            norm_val += expf(curr - max_val);
        }

        for (int col = 0; col < N; col++){
            out[row*N + col] = expf(inp[row*N + col] - max_val) / norm_val;
        }
    }
}

void run_online_softmax(float *inp, float *out, int M, int N){
    dim3 ThreadsPerBlock(1024);
    dim3 BlocksPerGrid((M + ThreadsPerBlock.x - 1) / ThreadsPerBlock.x);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    online_softmax_kernel<<<BlocksPerGrid, ThreadsPerBlock>>>(inp, out, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}