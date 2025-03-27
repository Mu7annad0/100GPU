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

__global__ void softmax_shared(float* __restrict__ inp, float* __restrict__ out, int M, int N) {
    // Shared memory allocation for reduction
    __shared__ float smem[1024];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Skip rows beyond matrix dimensions
    if (row >= M) return;
    float* input_row = inp + row * N;
    float* output_row = out + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // Compute local max and normalized values
    for (int i = tid; i < N; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    // Store local max in shared memory
    smem[tid] = local_max;
    __syncthreads();

    // Parallel reduction for row maximum
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] = fmax(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    // Get global row maximum
    float row_max = smem[0];
    __syncthreads();

    // Prepare normalized values in shared memory
    smem[tid] = local_norm * expf(local_max - row_max);
    __syncthreads();

    // Parallel reduction for normalization factor
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // Get global normalization factor
    float row_norm = smem[0];
    __syncthreads();

    // Compute final softmax values
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}
int main() {
    int M = 1024;
    int N = 32830;

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
    softmax_shared<<<gridSize, blockSize>>>(A_d, B_d, M, N);
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