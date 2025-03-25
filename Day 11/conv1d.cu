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
#define M_PI 3.14159265358979323846f

// CUDA kernel for 1D convolution with shared memory
__global__ void conv1d(float* input, float* filter, float* output, int N, int K, int M) {
    extern __shared__ float shared_input[];
    
    int B = blockDim.x;
    int start = blockIdx.x * B;
    int idx = start + threadIdx.x;
    
    // Load input into shared memory
    for (int i = threadIdx.x; i < B + K - 1; i += B) {
        int load_idx = start + i;
        if (load_idx < N) {
            shared_input[i] = input[load_idx];
        }
    }
    __syncthreads();
    
    // Compute convolution
    if (idx < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += shared_input[threadIdx.x + k] * filter[k];
        }
        output[idx] = sum;
    }
}

// CPU function for 1D convolution
void conv1d_cpu(float* input, float* filter, float* output, int N, int K, int M) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += input[i + k] * filter[k];
        }
        output[i] = sum;
    }
}

// Function to generate random numbers clamped between min and max
float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min) return min;
    if (num > max) return max;
    return num;
}

int main() {
    // Define dimensions
    int N = 1024;       // Input size
    int K = 5;          // Filter size
    int M = N - K + 1;  // Output size

    // Calculate sizes
    int size_input = N * sizeof(float);
    int size_filter = K * sizeof(float);
    int size_output = M * sizeof(float);

    // Allocate host memory
    float *input_h = (float*)malloc(size_input);
    float *filter_h = (float*)malloc(size_filter);
    float *output_h_gpu = (float*)malloc(size_output);  // GPU result
    float *output_h_cpu = (float*)malloc(size_output);  // CPU result

    // Initialize input and filter with sample data
    for (int i = 0; i < N; i++) {
        input_h[i] = random_normal_clamped(-5, 5);
    }
    for (int i = 0; i < K; i++) {
        filter_h[i] = random_normal_clamped(-1, 1); // Smaller range for filter
    }

    // Allocate device memory
    float *input_d, *filter_d, *output_d;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    // GPU memory allocation
    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&input_d, size_input));
    CUDA_CHECK(cudaMalloc(&filter_d, size_filter));
    CUDA_CHECK(cudaMalloc(&output_d, size_output));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> GPU allocation time: %f ms\n", ms);

    // Copy data to GPU
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(input_d, input_h, size_input, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(filter_d, filter_h, size_filter, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Host to device transfer time: %f ms\n", ms);

    // Define block and grid dimensions
    int block_size = 256;  // Threads per block
    int grid_size = CEIL_DIV(M, block_size);  // Number of blocks
    size_t shared_mem_size = (block_size + K - 1) * sizeof(float);  // Shared memory per block

    // Launch kernel
    cudaEventRecord(start);
    conv1d<<<grid_size, block_size, shared_mem_size>>>(input_d, filter_d, output_d, N, K, M);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    // Copy results back from GPU
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(output_h_gpu, output_d, size_output, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Device to host transfer time: %f ms\n", ms);

    // Compute convolution on CPU
    conv1d_cpu(input_h, filter_h, output_h_cpu, N, K, M);

    // Verify results
    float max_error = 0.0f;
    int error_count = 0;
    const float tolerance = 1e-5;
    for (int i = 0; i < M; i++) {
        float diff = fabs(output_h_gpu[i] - output_h_cpu[i]);
        if (diff > tolerance) {
            error_count++;
            if (diff > max_error) max_error = diff;
            if (error_count <= 5) {  // Print first 5 errors
                printf("Mismatch at index %d: GPU = %f, CPU = %f, Diff = %f\n",
                       i, output_h_gpu[i], output_h_cpu[i], diff);
            }
        }
    }
    if (error_count == 0) {
        printf("Verification passed: GPU and CPU results match within tolerance %e\n", tolerance);
    } else {
        printf("Verification failed: %d mismatches found, max error = %f\n", error_count, max_error);
    }

    // Free memory
    free(input_h);
    free(filter_h);
    free(output_h_gpu);
    free(output_h_cpu);
    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(filter_d));
    CUDA_CHECK(cudaFree(output_d));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}