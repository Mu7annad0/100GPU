#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))
#define BM 64
#define BK 8
#define BN 64
#define COARSE_FACTOR 8
#define M_PI 3.14159265358979323846f

__global__ void geluKernel(float* inp, float* out, int M, int N){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < N && row < M) {
        int idx = row * N + col;
        float x = inp[idx];
        out[idx] = x * 0.5 * (1 + tanhf(sqrt(2/M_PI) * (x + 0.044715 * x*x*x)));
    }
}

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

int main() {
    // Define matrix dimensions
    int M = 1024;
    int N = 1024;

    int size_A = M * N * sizeof(float);
    int size_B = M * N * sizeof(float);

    // Allocate host memory
    float *A_h = (float*)malloc(size_A);
    float *B_h = (float*)malloc(size_B);
    
    // Initialize input with sample data for matrix A (M x N)
    for (int i = 0; i < M*N; i++){
        A_h[i] = random_normal_clamped(-5, 5);
    }
    
    // Allocate device memory
    float *A_d, *B_d;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms=0.0f;

    // GPU memory allocation
    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&A_d, size_A));
    CUDA_CHECK(cudaMalloc(&B_d, size_B));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    // Copy data to GPU
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    // Define block and grid dimensions
    dim3 block_size(BM * BN / COARSE_FACTOR);
    dim3 grid_size(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    // Launch kernel
    cudaEventRecord(start);
    geluKernel<<<grid_size, block_size>>>(A_d, B_d, M, N);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);
    
    // Copy results back
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(B_h, B_d, size_B, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    // Free memory
    free(A_h);
    free(B_h);
    cudaFree(A_d);
    cudaFree(B_d);
    
    return 0;
}