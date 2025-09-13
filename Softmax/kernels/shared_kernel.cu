#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.cu"

__global__ void shared_softmax(float* inp, float* out, int N){

    __shared__ float shared_mem[1024];

    int tid = threadIdx.x;
    int row = blockIdx.x;

    if (row >= N) return;

    float* inp_row = inp + row * N;
    float* out_row = out + row * N;

    float norm_val = 0.0f;
    float local_max = -1 * INFINITY;

    for (int i = tid; i < N; i+=blockDim.x){
        float val = inp_row[i];
        if (val > local_max){
            norm_val = norm_val * expf(local_max - val);
            local_max = val;
        }
        norm_val += expf(val - local_max);
    }
    __syncthreads();

    shared_mem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2 ; stride > 0; stride /= 2){
        if (tid < stride){
            shared_mem[tid] = fmax(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    float global_max = shared_mem[0];
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride >>=1){
        if(tid < stride){
            shared_mem[tid] += shared_mem[tid+stride];
        }
        __syncthreads();
    }
    float global_norm = shared_mem[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        out_row[i] = expf(inp_row[i] - global_max) / global_norm;
    }
    
    
}