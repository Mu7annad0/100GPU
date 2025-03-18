#include <cuda_runtime.h>

#define BLUR_SIZE 1

// CUDA kernel for RGB image blur
__global__ void blurKernel(unsigned char* in, unsigned char* out, int width, int height, int channels) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < width && row < height){
        int pixR=0;
        int pixG=0;
        int pixB=0;
        int pixels=0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow){
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                int p = (curRow*width + curCol) * channels;

                if (curRow >= 0 && curRow < height && curCol >=0 && curCol < width){
                    pixR += in[p];
                    pixG += in[p+1];
                    pixB += in[p+2];
                    ++pixels;
                }
            }
        }

        int p = (row*width + col) * channels;
        out[p] = (unsigned char)(pixR/pixels);
        out[p+1] = (unsigned char)(pixG/pixels);
        out[p+2] = (unsigned char)(pixB/pixels);
    }
}

int main() {
    int height = 20;
    int width = 20;
    
    int size = 3 * height * width * sizeof(unsigned char);
    int channels = 3;

    // Allocate host memory
    unsigned char *input_h = (unsigned char*)malloc(size);
    unsigned char *output_h = (unsigned char*)malloc(size);
    
    // Initialize input with sample data
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int idx = (row * width + col) * channels;
            // Create a pattern with different values
            input_h[idx] = (row * 10) % 256;       // R varies with row
            input_h[idx+1] = (col * 10) % 256;     // G varies with column
            input_h[idx+2] = ((row+col) * 5) % 256; // B varies with both
        }
    }
    
    // Allocate device memory
    unsigned char *input_d, *output_d;
    cudaMalloc(&input_d, size);
    cudaMalloc(&output_d, size);
    
    // Copy input data to device
    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((int)ceil(width/16), (int)ceil(height/16));
    
    // Launch kernel
    blurKernel<<<gridDim, blockDim>>>(input_d, output_d, width, height, channels);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(output_h, output_d, size, cudaMemcpyDeviceToHost);

    // Free memory
    free(input_h);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);
    
    return 0;
}