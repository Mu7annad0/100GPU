#include <iostream>
#include <cuda_runtime.h>


#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}

#define M_PI 3.14159265358979323846f
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

__global__ void softmaxKernel(float* __restrict__ A, float* __restrict__ B, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M){
        float m = -1 * INFINITY;
        float L = 0.0f;

        for (int col = 0; col < N; col++){
            m = fmaxf(m, A[row * N + col]);
        }

        for (int col = 0; col < N; col++){
            L += expf(A[row * N + col] - m);
        }

        for (int col = 0; col < N; col++){
            B[row * N + col] = expf(A[row * N + col] - m) / L;
        }
    }
}

int main() {
    int M = 1024;
    int N = 512;

    int size = M*N;

    float* A_h = (float*)malloc(size*sizeof(float));
    float* B_h = (float*)malloc(size*sizeof(float));

    // Initialize input matrix with random values
    for (int i = 0; i < size; i++){
        A_h[i] = random_normal_clamped(-5, 5);
    }

    float *A_d, *B_d;

    int blockSize = 215;
    int gridSize = (M + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms=0.0f;

    // GPU memory allocation
    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&A_d, size*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B_d, size*sizeof(float)));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    // Copy data to GPU
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    // Run kernel
    cudaEventRecord(start);
    softmaxKernel<<<gridSize, blockSize>>>(A_d, B_d, M, N);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);

    // Copy results back
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(B_h, B_d, size * sizeof(float), cudaMemcpyDeviceToHost));
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