#include <cuda_runtime.h>
#include <iostream>
using namespace std;


// CUDA kernel for RGB to grayscale conversion
__global__ void rgbToGreyKernel(unsigned char* rgb, unsigned char* grey, int width, int height) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    
    if ((col < width) && (row < height)) {
        int grey_offset = row * width + col;
        int rgb_offset = grey_offset * 3;
        
        unsigned char r = rgb[rgb_offset];
        unsigned char g = rgb[rgb_offset + 1];
        unsigned char b = rgb[rgb_offset + 2];
        
        grey[grey_offset] = (0.21f * r + 0.72f * g + 0.07f * b);
    }
}

int main() {
    int height = 20;
    int width = 20;
    int input_size = 3 * height * width * sizeof(unsigned char);
    int output_size = height * width * sizeof(unsigned char);

    unsigned char *input_d, *output_d;
    unsigned char *input_h = (unsigned char*)malloc(input_size);
    unsigned char *output_h = (unsigned char*)malloc(output_size);
    
    // Initialize input with sample data
    for (int i = 0; i < height * width * 3; i += 3) {
        input_h[i] = 100;    // R
        input_h[i+1] = 150;  // G
        input_h[i+2] = 200;  // B
    }
    
    // Allocate device memory
    cudaMalloc(&input_d, input_size);
    cudaMalloc(&output_d, output_size);
    
    // Copy input data to device
    cudaMemcpy(input_d, input_h, input_size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    rgbToGreyKernel<<<gridDim, blockDim>>>(input_d, output_d, width, height);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(output_h, output_d, output_size, cudaMemcpyDeviceToHost);
    
    // Print a sample of the results
    cout << "Sample grayscale values:" << endl;
    for (int i = 0; i < 5; i++) {
        cout << "Pixel " << i << ": " << (int)output_h[i] << endl;
    }
    
    // Free memory
    free(input_h);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);
    
    return 0;
}