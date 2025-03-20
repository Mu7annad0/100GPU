#include <stdio.h>
#include <cuda_runtime.h>


#define M_PI 3.14159265358979323846f
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

// CUDA kernel for transposing a matrix
__global__ void transposeMatrixKernel(float* __restrict__ A, float* __restrict__ B, int M, int N) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < N && row < M) {
        B[col * M + row] = A[row * N + col];
    }
}

/*
Helper function to generate a clamped random number sampled from a
normal distribution with mean 0 and std 1
*/
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
    int M = 1024;
    int N = 1024;
    
    int a_size = M * N;
    int b_size = N * M;

    printf("Shape A: (%d, %d)\n", M, N);
    printf("Shape B (transposed): (%d, %d)\n", N, M);

    float* A = (float*)malloc(a_size * sizeof(float));
    float* B = (float*)malloc(b_size * sizeof(float));

    // Initialize matrix A with random values
    for (int i = 0; i < a_size; i++) {
        A[i] = random_normal_clamped(-10, 10);
    }

    // Print Matrix A before transposition
    // printf("Matrix A (Before Transposition):\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%6.2f ", A[i * N + j]);
    //     }
    //     printf("\n");
    // }

    float *Ad, *Bd;

    // Set up proper block and grid dimensions for 2D threads
    dim3 blockDim(32, 32);
    dim3 gridDim((int)ceil((float)N/32), (int)ceil((float)M/32));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&Ad, a_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Bd, b_size * sizeof(float)));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(Ad, A, a_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    cudaEventRecord(start);
    transposeMatrixKernel<<<gridDim, blockDim>>>(Ad, Bd, M, N);
    // Add error check after kernel launch
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(B, Bd, b_size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    // Print Matrix B after transposition
    // printf("Matrix B (After Transposition):\n");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < M; j++) {
    //         printf("%6.2f ", B[i * M + j]);
    //     }
    //     printf("\n");
    // }

    // Free memory
    free(A);
    free(B);
    cudaFree(Ad);
    cudaFree(Bd);
    
    return 0;
}
