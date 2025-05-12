#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}

#define M_PI 3.14159265358979323846  /* pi */
float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}

#define EPSILON 1e-6

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


int main() {
    int M = 1024;
    int N = 1024;

    // Allocate host memory
    float* inp_h = (float*)malloc(M * N * sizeof(float));
    float* out_h = (float*)malloc(M * N * sizeof(float));

    // Initialize input array with random values
    for (int i = 0; i < M * N; i++) {
        inp_h[i] = random_normal_clamped(-1.0f, 1.0f);
    }

    // Device pointers
    float *inp_d, *out_d;

    dim3 threadsPerBlock(1024); // 1024 rows
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x);


    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    // GPU memory allocation
    CUDA_CHECK(cudaMalloc(&inp_d, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_d, M * N * sizeof(float)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(inp_d, inp_h, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    cudaEventRecord(start);
    layer_norm<<<blocksPerGrid, threadsPerBlock>>>(inp_d, out_d, M, N);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(out_h, out_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Input matrix: \n");
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            printf("%f", inp_h[i*N+j]);
        }
        printf("\n");
    }

    printf("Output matrix: \n");
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            printf("%f", out_h[i*N+j]);
        }
        printf("\n");
    }

    // Free memory
    free(inp_h);
    free(out_h);
    cudaFree(inp_d);
    cudaFree(out_d);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}