#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32
__global__ void shuffled_softmax(float* inp, float* out, int M, int N){
    __shared__ float shared_mem[1024];

    int tid = threadIdx.x;
    int row = blockIdx.x;

    if (row >= M) return;

    float* inp_row = inp + row * N;
    float* out_row = out + row * N;

    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // -------------------------
    // Phase 1: Compute local max and partial norm for this thread
    // -------------------------
    for (int i = tid; i < N; i+=blockDim.x){
        float val = inp_row[i];
        if (val > local_max){
            local_norm = local_norm * expf(local_max - val);
            local_max = val;
        }
        local_norm += expf(val - local_max);
    }
    __syncthreads();

    // -------------------------
    // Phase 2a: Warp-level reduction to get row-wise max
    // -------------------------
    float val = local_max;
    for (int offset = WARP_SIZE/2; offset > 0 ; offset /= 2){
        val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }

    // Block-wide reduction for row max (multiple warps case)
    if (blockDim.x > WARP_SIZE){
        if (tid % WARP_SIZE == 0){
            shared_mem[tid/WARP_SIZE] = val;
        }
        __syncthreads();
        if (tid < WARP_SIZE){
            // reduce the per-warp results
            val = (tid < ((blockDim.x + WARP_SIZE - 1) / WARP_SIZE)) ? shared_mem[tid] : -INFINITY;
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
                val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
            }
            if (tid == 0){
                shared_mem[0] = val;
            }
        }
    }
    else{
        if (tid == 0){
            shared_mem[0] = val;
        }
    }
    __syncthreads();
    float row_max = shared_mem[0];
    __syncthreads();

    // -------------------------
    // Phase 2b: Warp/block reduction to get row-wise normalization factor
    // -------------------------
    val = local_norm * expf(local_max - row_max);

    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    if (blockDim.x > WARP_SIZE){
        if (tid % WARP_SIZE == 0){
            shared_mem[tid/WARP_SIZE] = val;
        }
        __syncthreads();
        
        if (tid < WARP_SIZE){
            val = (tid < ((blockDim.x + WARP_SIZE - 1) / WARP_SIZE)) ? shared_mem[tid] : -INFINITY;
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2){
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            }
            if (tid == 0){
                shared_mem[0] = val;
            }
        }
    }
    else{
        if (tid == 0){
            shared_mem[0] = val;
        }
    }
    __syncthreads();
    float row_norm = shared_mem[0];
    __syncthreads();

    // -------------------------
    // Phase 3: Write normalized softmax outputs
    // -------------------------
    for (int i = tid; i < N; i += blockDim.x) {
        out_row[i] = expf(inp_row[i] - row_max) / row_norm;
    }
}


void run_shuffled_softmax(float* inp, float* out, int M, int N){
    dim3 ThreadsPerBlock(1024);
    dim3 BlocksPerGrid(M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    shuffled_softmax<<<BlocksPerGrid, ThreadsPerBlock>>>(inp, out, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}