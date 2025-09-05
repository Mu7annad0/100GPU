#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>

#include "kernels/naive_matmul/thread_for_out_col.cu"

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
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}

#define EPSILON 1e-6

int main(){
    int M = 4;
    int K = 3;
    int N = 3;

    // Allocate host memory
    float* A_h = (float*)malloc(M * K * sizeof(float));
    float* B_h = (float*)malloc(K * N * sizeof(float));
    float* C_h = (float*)malloc(M * N * sizeof(float));

    // Initialize input array with random values
    for (int i = 0; i < M * K; i++) {
        A_h[i] = random_normal_clamped(-1.0f, 1.0f);
    }
    for (int i = 0; i < K * N; i++) {
        B_h[i] = random_normal_clamped(-1.0f, 1.0f);
    }

    // Device pointers
    float *A_d, *B_d, *C_d;

    CUDA_CHECK(cudaMalloc(&A_d, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B_d, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C_d, M * N * sizeof(float)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(A_d, A_h, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, K * N * sizeof(float), cudaMemcpyHostToDevice));

    run_matmul_thread_one_col(A_d, B_d, C_d, M, K, N);

    CUDA_CHECK(cudaMemcpy(C_h, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Input matrix A: \n");
    for(int i=0; i<M; i++){
        for(int j=0; j<K; j++){
            printf("%f", A_h[i*K+j]);
        }
        printf("\n");
    }

    printf("Input matrix B: \n");
    for(int i=0; i<K; i++){
        for(int j=0; j<N; j++){
            printf("%f", B_h[i*N+j]);
        }
        printf("\n");
    }

    printf("Output matrix: \n");
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            printf("%f", C_h[i*N+j]);
        }
        printf("\n");
    }

    // Free memory
    free(A_h);
    free(B_h);
    free(C_h);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}
