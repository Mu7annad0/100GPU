#include <stdio.h>
#include <cuda_runtime.h>


#define TILE_WIDTH 16

__global__ void tiled_matmul_cornertuning(float *A, float *B, float *C, int M, int K, int N){
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float val = 0.0;

    for (int phase = 0; phase < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++phase){
        // ---------------- Load A tile into shared memory ----------------
        // Each thread loads one element of A from global memory

        int a_col = phase*TILE_WIDTH + tx;
        if (row < M && a_col < K) {
            // Access is row-major: consecutive threads load consecutive elements
            A_shared[ty][tx] = A[row*K + a_col];
        } else {
            A_shared[ty][tx] = 0.0f;
        }

        /// ---------------- Load B tile into shared memory ----------------
        // Each thread loads one element of B, but we TRANSPOSE it when storing
        // This ensures global memory accesses are coalesced
        int b_row = phase*TILE_WIDTH + ty;
        if (b_row < K && col < N) {
            // Store transposed: (tx, ty) instead of (ty, tx)
            B_shared[tx][ty] = B[b_row*N + col];
        } else {
            B_shared[tx][ty] = 0.0f;
        }

        __syncthreads();

        // ---------------- Compute partial results ----------------
        // Multiply the row of A_shared by the column of B_shared
        // Note: we use the transposed version of B_shared here
        for (int j = 0; j < TILE_WIDTH; ++j) {
            val += A_shared[ty][j] * B_shared[tx][j];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
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

    tiled_matmul_cornertuning<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}    
