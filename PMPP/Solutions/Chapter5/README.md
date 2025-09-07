## Exercise 1
Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.

### Solution
In matrix Addition each thread computes on element of the output matrix. 
`C[i,j]=A[i,j]+B[i,j]` <br>
[1  2] + [5 6] = [6 8] <br>
[3  4] + [7 8] = [10 12] <br>
And shared memory is usefull if multiple threads in a block access the same data multiple times (So loaded once from global memory and stored in shared memory). <br>
But in matrix addition each thread access one element of A and one element of B, **so shared memory is not usefull.**

## Exercise 3
What type of incorrect execution behavior can happen if one forgot to use one or both __syncthreads() in the kernel of Fig. 5.9?


### Solution
The first Sync is after loading the tiles from the global memory to the Shared Memory, so if we forgot to use it, we will have a race condition, where threads will try to access the shared memory before it is loaded.

The second Sync is after the dot product computation, so if we forgot to use it, we will have a race condition, where threads will starting to overwrite the values Mds and Nds when we are still reading from them.

If both are forgotten, threads will try to access the shared memory before it is loaded and starting to overwrite the values Mds and Nds when we are still reading from them.

## Exercise 4
Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain your answer.

### Solution
While registers are very fast, they are private to each thread and cannot be accessed by other threads. This means that any value stored in a register is only useful for the thread that owns it.

In contrast, shared memory is accessible by all threads within a block. Using shared memory allows multiple threads to reuse the same data without repeatedly fetching it from global memory. This makes shared memory valuable for inter-thread communication and for reducing redundant global memory accesses when data needs to be shared among threads.

## Exercise 5
For our tiled matrix-matrix multiplication kernel, if we use a 32x32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?

### Solution
Let's assume M is of size m x n and N is of size n x o.

Let's think step by step. For row 0 of matrix P, we used to load the entire row 0 of matrix M for every single value in this row (n loads). Now we load the values once from global memory and put them in the first row of the tile 'Mds.' We just reduced the loads from global memory from n to n/32, so the reduction is 32x. The same thing is repeated for every single row in this matrix; e.g., assuming the M and N matrices are both '32x32,' we would go from 32 loads to 1 load. This is repeated for every row of M and for every column of N.

##  Exercise 6
Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?

### Solution
We have 1000 blocks -- and for each each block has 512 threads, and the varibale will be intialized and stored for every single one of them, so we have 1000 * 512 = **512000** versions of the variable.

## Exercise 7
In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?

### Solution 
A shared Memory variable is shared per block, so we will have **1000** versions of the variable.

## Exercise 8
Consider performing a matrix multiplication of two input matrices with dimensions N x N. How many times is each element in the input matrices requested from global memory when: <br>
**A. There is no tiling?** <br>
**B. Tiles of size TxT are used?**

### Solution
- **A**. There is no tiling? <br>
each input element is read **N** times from global memory. 
 
- **B**. Tiles of size TxT are used? <br>
each input element is read **N/T** times from global memory.

## Exercise 9
A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory- bound. <br>
**A. Peak FLOPS=200 GFLOPS, peak memory bandwidth=100 GB/second** <br>
**B. Peak FLOPS=300 GFLOPS, peak memory bandwidth=250 GB/second**

### Solution
To determine if the kernel is Compute Bound or Memory Bound, we need to calculate its artimitic intensity in FLOPs per Byte, and compare it to the machine balance (Peack flops / peak memory bandwidth).

- If AI < machine balance, the kernel is memory-bound.
- If AI > machine balance, the kernel is compute-bound.

The kernel performs 36 floating-point operation per thread, and seven 32-bit (4 Bytes) global memory accesses per thread, so it transfers 7 × 4 = 28 bytes per thread. Thus, AI = 36 FLOPs / 28 bytes ≈ 1.286 FLOPs/byte.

**A.** Machine balance = 200 GFLOPS / 100 GB/s = 2 FLOPs/byte. Since 1.286 < 2, the kernel is **memory-bound**. <br>
**B.** Machine balance = 300 GFLOPS / 250 GB/s = 1.2 FLOPs/byte. Since 1.286 > 1.2, the kernel is **compute-bound**.

## Exercise 10
To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.

```cpp
01  dim3 blockDim(BLOCK_WIDTH,BLOCK_WIDTH);
02  dim3 gridDim(A_width/blockDim.x,A_height/blockDim.y);
03  BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

04  __global__ void
05  BlockTranspose(float* A_elements, int A_width, int A_height)
06  {
07      __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

08      int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
09      baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

10     blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

11     A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
12 }
```

**A. Out of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will this kernel function execute correctly on the device?**

The kernel will only execute correctly for **BLOCK_WIDTH = 1** <br>
For any larger value (2–20), it requires a synchronization barrier (__syncthreads()) to ensure correctness.

**B. If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_SIZE values.**

The incorrect execution comes from a race condition in shared memory:
- In line 10, each thread writes its element into blockA[ty][tx].
- In line 11, threads immediately start reading from blockA[tx][ty].
Without a barrier, some threads may read before others have written, leading to undefined/incorrect results. <br>

This problem appears whenever more than one thread per block is active. With BLOCK_WIDTH = 1 only one thread exists, so there’s no race. <br>
Insert a synchronization barrier after all threads finish writing to shared memory, before any thread reads from it.

Corrected kernel:
```cpp
__global__ void
BlockTranspose(float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_WIDTH + threadIdx.y) * A_width;

    // Load element into shared memory
    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

    // Ensure all threads have finished writing before any reads occur
    __syncthreads();

    // Write transposed element back
    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```

## Exercise 11
Consider the following CUDA kernel and the corresponding host function that calls it:
```cpp
01  __global__ void foo_kernel(float* a, float* b) {
02      unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
03      float x[4];
04      __shared__ float y_s;
05      __shared__ float b_s[128];
06      for(unsigned int j = 0; j < 4; ++j) {
07          x[j] = a[j*blockDim.x*gridDim.x + i];
08      }
09      if(threadIdx.x == 0) {
10         y_s = 7.4f;
11     }
12     b_s[threadIdx.x] = b[i];
13     __syncthreads();
14     b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]
15             + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
16 }
17 void foo(int* a_d, int* b_d) {
18     unsigned int N = 1024;
19     foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
20 }
```

**A. How many versions of the variable i are there?**
The variable i is considered as a local variable accesed by all the threads in the blocks, so we need to calculate the number of threads in the blocks, and multiply it by the number of blocks, so we have **128 * (1024 + 128 - 1)/128 = 1024** versions of the variable.

**B. How many versions of the array x[] are there?**
Also this variable is like the above considered as a local var, so there will be **1024** versions of the variable.

**C. How many versions of the variable y_s are there?**
The variable y_s is a shared memory variable and it's only shared per block, so we need to calculate the number of blocks, ((1024 + 128 - 1)/128 = 8) and so we have **8** versions of the variable.

**D. How many versions of the array b_s[] are there?**
Same as the variable in C, we will have 8 versions of the b_s[] variable.

**E. What is the amount of shared memory used per block (in bytes)?**
The shared memory variables is y_s (1 float) & b_s (array of 128 floats)
- y_s = 1 x 4 bytes = 4 bytes
- b_s = 128 x 4 bytes = 512 bytes

So the amount of shared memory used per block is **516 bytes**

**F.  What is the floating-point to global memory access ratio of the kernel (in OP/B)?**

FLOPs per thread:
`2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3] + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128]`
we have 6 multiplications, 4 additions, so 10 FLOPs per thread

Global Mmeory acceses <br>
- In line 07 loads 4 floats from global memory. <br>
- In line 12 loads 1 float from global memory. <br>
- In line 14 stores 1 float to global memory. <br>

So we have 6 global memory acceses and each access = 4 bytes
Total = 6 x 4 = 24 bytes.

So the floating-point to global memory access ratio of the kernel (in OP/B) is **10/24 = 0.4167 FLOPs/byte**.


## Exercise 12
Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64K (65,536) registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

**A. The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/block**
- Registers per block: ⌊27*64⌋ = 1728 registers
- Maximum number of blocks according to registers: ⌊65536/1728⌋ = 37 blocks
- Maximum number of blocks according to shared memory: ⌊96*1024/4096⌋ = 24 blocks
- Maximum number of blocks according to threads: ⌊2048/64⌋ = 32 blocks

So the effictive number of blocks is 24 blocks <br>
Number of threads: ⌊24*64⌋ = 1536 threads   <br>
The occupancy: 1536/2048 = **75%**

**B. The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/block.**
- Registers per block: ⌊31*256⌋ = 7,936 registers
- Maximum number of blocks according to registers: ⌊65536/7936⌋ = 8 blocks
- Maximum number of blocks according to shared memory: ⌊96*1024/8192⌋ = 12 blocks
- Maximum number of blocks according to threads: ⌊2048/256⌋ = 8 blocks

So the effictive number of blocks is 8 blocks <br>
Number of threads: ⌊8*256⌋ = 2048 threads <br>
The occupancy: 2048/2048 = **100%**
