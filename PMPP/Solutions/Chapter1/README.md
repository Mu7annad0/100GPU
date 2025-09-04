## Exercise 1
If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?
- A. i=threadIdx.x + threadIdx.y;
- B. i=blockIdx.x + threadIdx.x;
- C. i=blockIdx.x blockDim.x + threadIdx.x;
- D. i=blockIdx.x threadIdx.x;

### Solution
**(B)** Each block has blockDim.x threads, so to get a unique global index we offset by blockIdx.x * blockDim.x and add the local thread index.


## Exercise 2
Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?
- A. i=blockIdx.x blockDim.x + threadIdx.x +2;
- B. i=blockIdx.x*threadIdx.x*2;
- C. i=(blockIdx.x*blockDim.x + threadIdx.x)*2;
- D. i=blockIdx.x*blockDim.x*2 + threadIdx.x;

### Solution
**(C)** Each thread handles 2 elements, so we first compute the global thread index, then multiply by 2 to get the starting position.

## Exercise 3
We want to use each thread to calculate two elements of a vector addition. Each thread block processes 2*blockDim.x consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?
- A.  i=blockIdx.x*blockDim.x + threadIdx.x +2;
- B. i=blockIdx.x*threadIdx.x*2;
- C. i=(blockIdx.x*blockDim.x + threadIdx.x)*2;
- D. i=blockIdx.x*blockDim.x*2 + threadIdx.x;

### Solution
**(D)** Each block handles 2 * blockDim.x elements, so we offset by that, then within the block each thread starts at its own threadIdx.x.

## Exercise 4
For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?
- A. 8000
- B. 8196 
- C. 8192 
- D. 8200

### Solution
**(C)** Blocks = ceil(8000/1024) = 8; total threads = 8 × 1024 = 8192, the minimum covering all 8000 elements.

## Exercise 5
If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the cudaMalloc call?
- A. n
- B. v
- C. n * sizeof(int)
- D. v * sizeof(int)

### Solution
**(D)** cudaMalloc needs the size in bytes, so we multiply the number of integers v by sizeof(int).

## Exercise 6
If we want to allocate an array of n floating-point elements and have a floating-point pointer variable A_d to point to the allocated memory, what would be an appropriate expression for the first argument of the cudaMalloc () call?
- A. n
- B. (void *) A_d
- C. *A_d
- D. (void**)(&A_d)

### Solution
**(D)** cudaMalloc requires the address of the device pointer, so we pass &A_d cast to (void**).

## Exercise 7
If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer to element 0 of the source array) to device array A_d (A_d is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?
- A. cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);
- B. cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceTHost); 
- C. cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice); 
- D. cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);

### Solution
**(C)** cudaMemcpy takes destination first, then source, then size in bytes, then the direction flag.

## Exercise 8
How would one declare a variable err that can appropriately receive the returned value of a CUDA API call?
- A. int err;
- B. cudaError err;
- C. cudaError_t err; 
- D. cudaSuccess_t err;

### Solution
**(C)** CUDA API calls return a value of type cudaError_t, which is used to check for errors.

## Exercise 9
Consider the following CUDA kernel and the corresponding host function that calls it:
```cpp
01 __global__ void foo_kernel(float* a, float* b, unsigned int N) {
02     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
03     
04     if (i < N) {
05         b[i] = 2.7f * a[i] - 4.3f;
06     }
07 }
08 
09 void foo(float* a_d, float* b_d) {
10     unsigned int N = 200000;
11     foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
12 }
```
**A. What is the number of threads per block?** <br>
128 (from the kernel launch, second parameter).

**B. What is the number of threads in the grid?** <br>
200064 (1563 blocks × 128 threads).

**C. What is the number of blocks in the grid?** <br>
1563  (calculated as (N + 128 - 1) / 128).

**D. What is the number of threads that execute the code on line 02** <br>
200,064 all launched threads compute their index i

**E. What is the number of threads that execute the code on line 04** <br>
200,000  only threads with i < N (valid elements).

## Exercise 10
A new summer intern was frustrated with CUDA. He has been complaining that CUDA is very tedious. He had to declare many functions that he plans to execute on both the host and the device twice, once as a host function and once as a device function. What is your response?

### Solution
He doesn’t need to declare the same function twice — CUDA provides the qualifier __host__ __device__, which lets a function be compiled for both host and device.