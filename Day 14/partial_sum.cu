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

__global__ void partialSumKernel(int *inp, int *out, int n) {
    extern __shared__ int sharedMemory[];
    
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x*2 + tid;
    if (index < n) {
        // Load input into shared memory
        sharedMemory[tid] = inp[index]+inp[index+blockDim.x];
        __syncthreads();
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int temp = 0;
            if (tid >= stride) {
                temp = sharedMemory[tid - stride];
            }
            __syncthreads();
            sharedMemory[tid] += temp;
            __syncthreads();
        }
        // Write result to global memory
        out[index] = sharedMemory[tid];
    }
}


int main() {
    int N = 1024;  // Total number of elements

    // Allocate host memory
    int* inp_h = (int*)malloc(N * sizeof(int));
    int* out_h = (int*)malloc(N * sizeof(int));

    // Initialize input array
    for (int i = 0; i < N; i++) {
        inp_h[i] = i + 1;  // Example initialization
    }

    // Device pointers
    int *inp_d, *out_d;

    int blockSize = 256;  // Typical block size
    int gridSize = (N + 2*blockSize - 1) / (2*blockSize);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    // GPU memory allocation
    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&inp_d, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&out_d, N * sizeof(int)));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    // Copy data to GPU
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(inp_d, inp_h, N * sizeof(int), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    // Determine shared memory size
    size_t sharedMemSize = blockSize * sizeof(int);

    // Run kernel
    cudaEventRecord(start);
    partialSumKernel<<<gridSize, blockSize, sharedMemSize>>>(inp_d, out_d, N);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);

    // Copy results back
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(out_h, out_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    // Optionally, print some results
    printf("First few output elements:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", out_h[i]);
    }
    printf("\n");

    // Free memory
    free(inp_h);
    free(out_h);
    cudaFree(inp_d);
    cudaFree(out_d);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}