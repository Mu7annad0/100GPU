#include <iostream>
#include <cuda_runtime.h>

__global__ void layer_norm(float* input, float* output, int M, int N){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M){
        float mean = 0.0;
        float var = 0.0;

        for (int col = 0; col < N; col++){
            int idx = row * N + col;
            mean += input[idx];
        }
        mean /= N;

        for (int col = 0; col < N; col++){
            int idx = row * N + col;
            var += (input[idx] - mean) * (input[idx] - mean);
        }
        var /= N;

        float std = sqrtf(var + EPSILON);

        for (int col = 0; col < N; col++){
            int idx = row * N + col;
            output[idx] = (input[idx] - mean) / std;
        }
    }
}

void run_layer_norm(float * input, float * output, int M, int N){

    dim3 threadsPerBlock(1024); // 1024 rows
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    layer_norm<<<blocksPerGrid, threadsPerBlock>>>(input, output, M, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}    
