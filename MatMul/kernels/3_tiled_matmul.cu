#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define COARSE_FACTOR 4
__global__ void threadcoalesced_tiled_matmul(float *A, float *B, float *C, int M, int K, int N){
    
    // Shared memory for tile of A
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row this thread is responsible for
    int row = by * TILE_WIDTH + ty;
    
    // Starting column for this thread's coarse elements
    // Each thread computes COARSE_FACTOR elements horizontally
    int col_start = bx * TILE_WIDTH * COARSE_FACTOR + tx;
    
    float Pvalues[COARSE_FACTOR];
    for (int i = 0; i < COARSE_FACTOR; i++){
        Pvalues[i] = 0.0;
    }
    
    for (int phase = 0; phase < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++phase){
        
        // ===== LOAD A TILE =====
        // Each thread loads one element of A tile
        int a_col = phase * TILE_WIDTH + tx;
        if (row < M && a_col < K){
            A_shared[ty][tx] = A[row * K + a_col];
        } else {
            A_shared[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // ===== PROCESS COARSE_FACTOR B TILES =====
        for (int c = 0; c < COARSE_FACTOR; ++c){
            
            // Load B tile for this coarse iteration
            int b_row = phase * TILE_WIDTH + ty;
            int b_col = col_start + c * TILE_WIDTH;
            
            if (b_row < K && b_col < N){
                B_shared[ty][tx] = B[b_row * N + b_col];
            } else {
                B_shared[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            for (int k = 0; k < TILE_WIDTH; ++k){
                Pvalues[c] += A_shared[ty][k] * B_shared[k][tx];
            }
            
            __syncthreads();
        }
    }
    
    for (int c = 0; c < COARSE_FACTOR; ++c){
        int col = col_start + c * TILE_WIDTH;
        if (row < M && col < N){
            C[row * N + col] = Pvalues[c];
        }
    }
}

void run_tiled_matmul(float * A, float * B, float * C, int M, int K, int N){

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH * COARSE_FACTOR - 1) / (TILE_WIDTH * COARSE_FACTOR),
                   (M + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    cudaEventRecord(start);

    threadcoalesced_tiled_matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}