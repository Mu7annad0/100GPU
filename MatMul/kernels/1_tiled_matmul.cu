#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
__global__ void tiled_matmul(float *A, float *B, float *C, int M, int K, int N){
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float val = 0.0;

    // Loop over all "phases" (tiles) of A and B needed to compute C(row, col)
    for (int phase = 0; phase < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++phase){

        // Load tile from A into shared memory
        int a_col = phase*TILE_WIDTH + tx;
        if (row < M && a_col < K){
            A_shared[ty][tx] = A[row*K + a_col];
        } else{
            A_shared[ty][tx] = 0.0f; // pad with zero if outside matrix
        }

        // Load tile from B into shared memory
        int b_row = phase*TILE_WIDTH + ty;
        if (b_row < K && col < N){
            B_shared[ty][tx] = B[b_row*N + col];
        } else {
            B_shared[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Each thread computes a partial dot product for its element of C
        for (int j = 0 ; j < TILE_WIDTH ; ++j){
            val += A_shared[ty][j] * B_shared[j][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N){
        C[row*N + col] = val;
    }
}

void run_tiled_matmul(float * A, float * B, float * C, int M, int K, int N){

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1)/TILE_WIDTH ,(M + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    tiled_matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}    
