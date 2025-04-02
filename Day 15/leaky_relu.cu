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

__global__ void leaky_relu(const float *input, float *output, float alpha, int m, int n) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < n && row < m) {
        int idx = row * n + col;
        float inp = input[idx];
        output[idx] = inp > 0.0f ? inp : inp * alpha;
    }
}

int main() {
    int M = 1024;
    int N = 1024;
    float alpha = 0.01f;

    // Allocate host memory
    float* inp_h = (float*)malloc(M * N * sizeof(float));
    float* out_h = (float*)malloc(M * N * sizeof(float));

    // Initialize input array with random values
    for (int i = 0; i < M * N; i++) {
        inp_h[i] = random_normal_clamped(-1.0f, 1.0f);
    }

    // Device pointers
    float *inp_d, *out_d;

    // Define block and grid dimensions for 2D kernel
    dim3 blockDim(64, 64);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    // GPU memory allocation
    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&inp_d, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_d, M * N * sizeof(float)));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    // Copy data to GPU
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(inp_d, inp_h, M * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    // Run kernel
    cudaEventRecord(start);
    leaky_relu<<<gridDim, blockDim>>>(inp_d, out_d, alpha, M, N);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);

    // Copy results back
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(out_h, out_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    // Print some results for verification
    printf("Sample input and output values:\n");
    printf("Index\tInput\t\tOutput\n");
    for (int i = 0; i < 10; i++) {
        printf("%d\t%f\t%f\n", i, inp_h[i], out_h[i]);
    }

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