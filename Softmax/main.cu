#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>

#include "kernels/naive_softmax.cu"
#include "utils.cuh"


int main(){
    int M = 1024;
    int N = 32768;

    float *inp_h = (float*)malloc(M * N * sizeof(float));
    float *out_h = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * N; i++){
        inp_h[i] = random_normal_clamped(-1.0f, 1.0f);
    }

    float *inp_d, *out_d;
    CUDA_CHECK(cudaMalloc(&inp_d, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_d, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(inp_d, inp_h, M * N * sizeof(float), cudaMemcpyHostToDevice));

    run_naive_softmax(inp_d, out_d, M, N);

    CUDA_CHECK(cudaMemcpy(out_h, out_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory
    free(inp_h);
    free(out_h);
    cudaFree(inp_d);
    cudaFree(out_d);

    return 0;
}