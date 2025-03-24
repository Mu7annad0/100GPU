#include <cuda_runtime.h>
#include <stdio.h>


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
#define TILE_SIZE 32
#define BM 64
#define BK 8
#define BN 64
#define COARSE_FACTOR 8

__global__ void matrixMultiSharedKernel(float* A, float* B, float* C, int M, int N, int K) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int by = blockIdx.y;
    int bx = blockIdx.x;

    // indices of C[row, col]
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // tile that will be loaded by THIS block
    __shared__ float a_smem[TILE_SIZE][TILE_SIZE];
    __shared__ float b_smem[TILE_SIZE][TILE_SIZE];

    // final dot product sum
    float acc = 0.f;

    // THIS block will loop over the tiles in common dimension
    for (int tile_num = 0; tile_num < CEIL_DIV(K, TILE_SIZE); tile_num++) {
        int offset = tile_num * TILE_SIZE;

        // out of bounds check
        // same row, different column for A
        if (row < M && (offset + tx) < N)
            a_smem[ty][tx] = A[row * N + offset + tx];
        else
            a_smem[ty][tx] = 0.f;

        // different row, same column for B (N×K)
        if ((offset + ty) < N && col < K)
            b_smem[ty][tx] = B[(offset + ty) * K + col];
        else
            b_smem[ty][tx] = 0.f;
        __syncthreads();

        // dot product and accumulate
        for (int i = 0; i < TILE_SIZE; i++) {
            acc += a_smem[ty][i] * b_smem[i][tx];
        }
        __syncthreads();
    }

    // write the final output after looping over all tiles
    if (row < M && col < K) {
        C[row * K + col] = acc;
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

int main() {
    // Define matrix dimensions
    int M = 1024;
    int N = 1024;
    int K = 1024;

    int size_A = M * N * sizeof(float);
    int size_B = N * K * sizeof(float);
    int size_C = M * K * sizeof(float);

    // Allocate host memory
    float *A_h = (float*)malloc(size_A);
    float *B_h = (float*)malloc(size_B);
    float *C_h = (float*)malloc(size_C);
    
    // Initialize input with sample data for matrix A (M x N)
    for (int i = 0; i < M*N; i++){
        A_h[i] = random_normal_clamped(-5, 5);
    }

    for (int i = 0; i < N*K; i++){
        B_h[i] = random_normal_clamped(-5, 5);
    }
    
    
    // Allocate device memory
    float *A_d, *B_d, *C_d;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms=0.0f;

    // GPU memory allocation
    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&A_d, size_A));
    CUDA_CHECK(cudaMalloc(&B_d, size_B));
    CUDA_CHECK(cudaMalloc(&C_d, size_C));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    // Copy data to GPU
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    
    // Define block and grid dimensions
    dim3 block_size(BM * BN / COARSE_FACTOR);
    dim3 grid_size(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    
    // Launch kernel
    cudaEventRecord(start);
    matrixMultiSharedKernel<<<grid_size, block_size>>>(A_d, B_d, C_d, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);
    
    // Copy results back
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);


    // Free memory
    free(A_h);
    free(B_h);
    free(C_h);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
    return 0;
}