# PMPP Book Exercises - Solutions

## Exercise 1
Consider the following CUDA kernel and the corresponding host function that calls it:
```cpp
01 __global__ void foo_kernel(int* a, int* b) {
02     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
03     if(threadIdx.x < 40 || threadIdx.x >= 104) {
04         b[i] = a[i] + 1;
05     }
06     if(i%2 == 0) {
07         a[i] = b[i]*2;
08     }
09     for(unsigned int j = 0; j < 5 - (i%3); ++j) {
10         b[i] += j;
11     }
12 }
13 void foo(int* a_d, int* b_d) {
14     unsigned int N = 1024;
15     foo_kernel <<< (N + 128 - 1)/128, 128 >>>(a_d, b_d);
16 }
```

**A. What is the number of warps per block?**  
With 128 threads per block and 32 threads per warp, we have 128 ÷ 32 = **4** warps per block.

**B. What is the number of warps in the grid?**  
Number of blocks = (1024 + 128 - 1) ÷ 128 = 8 blocks  
Total warps = 8 blocks × 4 warps/block = **32** warps in the grid.

**C. For the statement on line 04:**  
The condition is `(threadIdx.x < 40 || threadIdx.x >= 104)`, meaning threads 0-39 and 104-127 execute this statement.

- **i. How many warps in the grid are active?**  
  - Warp 0 (threads 0-31): All threads active (all < 40)
  - Warp 1 (threads 32-63): Threads 32-39 active (8 threads)
  - Warp 2 (threads 64-95): No threads active (none < 40 or ≥ 104)
  - Warp 3 (threads 96-127): Threads 104-127 active (24 threads)
  
  Per block: 3 warps are active  
  Total active warps in grid: 8 × 3 = **24** warps

- **ii. How many warps in the grid are divergent?**  
  - Warp 1: Divergent (some threads active, some inactive)
  - Warp 3: Divergent (some threads active, some inactive)
  
  Per block: 2 warps are divergent  
  Total divergent warps in grid: 8 × 2 = **16** warps

- **iii. What is the SIMD efficiency (in %) of warp 0 of block 0?**  
  All 32 threads in warp 0 are active.  
  SIMD efficiency = 32/32 × 100 = **100%**

- **What is the SIMD efficiency (in %) of warp 1 of block 0?**  
  Only threads 32-39 are active (8 threads out of 32).  
  SIMD efficiency = 8/32 × 100 = **25%**

- **iv. What is the SIMD efficiency (in %) of warp 3 of block 0?**  
  Only threads 104-127 are active (24 threads out of 32).  
  SIMD efficiency = 24/32 × 100 = **75%**

**D. For the statement on line 07:**  
The condition is `(i%2 == 0)`, meaning threads with even global indices execute this statement.

- **i. How many warps in the grid are active?**  
  **All 32 warps** are active because every warp contains both even and odd indexed threads.

- **ii. How many warps in the grid are divergent?**  
  **All 32 warps** are divergent because within each warp, some threads have even indices (active) and some have odd indices (inactive).

- **iii. What is the SIMD efficiency (in %) of warp 0 of block 0?**  
  Half of the threads (those with even indices) are active.  
  SIMD efficiency = 16/32 × 100 = **50%**

## Exercise 2
For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

### Solution
Number of blocks = ⌈2000/512⌉ = ⌈3.91⌉ = 4 blocks  
Total threads = 4 × 512 = **2048** threads

## Exercise 3
For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?

### Solution
Total warps = 2048 ÷ 32 = 64 warps  
The boundary condition affects threads with indices ≥ 2000.  
- Threads 0-1999: Active (within bounds)
- Threads 2000-2047: Inactive (out of bounds)

The last active thread is at index 1999, which is in warp 62 (since 1999 ÷ 32 = 62.46).  
Warp 62 contains threads 1984-2015, where threads 1984-1999 are active and threads 2000-2015 are inactive.  
Warp 63 contains threads 2016-2047, all of which are inactive.

Therefore, **1 warp** (warp 62) experiences divergence.

## Exercise 4
Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9. What percentage of the threads' total execution time is spent waiting for the barrier?

### Solution
All threads must wait for the slowest thread (3.0 μs) to reach the barrier.

Waiting times for each thread:
- Thread 1: 3.0 - 2.0 = 1.0 μs
- Thread 2: 3.0 - 2.3 = 0.7 μs  
- Thread 3: 3.0 - 3.0 = 0.0 μs
- Thread 4: 3.0 - 2.8 = 0.2 μs
- Thread 5: 3.0 - 2.4 = 0.6 μs
- Thread 6: 3.0 - 1.9 = 1.1 μs
- Thread 7: 3.0 - 2.6 = 0.4 μs
- Thread 8: 3.0 - 2.9 = 0.1 μs

Total waiting time = 4.1 μs  
Total execution time = 8 × 3.0 = 24.0 μs  
Percentage waiting = 4.1/24.0 × 100 = **17.08%**

## Exercise 5
A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the `__syncthreads()` instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain.

### Solution
This is **not a good idea** for several reasons:

1. **Memory consistency**: While threads in a warp execute in lock-step (SIMT), this doesn't guarantee memory consistency. Memory operations may complete at different times due to caching, memory hierarchy, and hardware optimizations.

2. **Compiler optimizations**: The compiler may reorder instructions in ways that break the assumed synchronization, even within a warp.

3. **Future compatibility**: NVIDIA does not guarantee that warp size will remain 32 in future architectures. Code should be written to be portable across different hardware generations.

4. **Code clarity and maintainability**: Using `__syncthreads()` makes synchronization requirements explicit, improving code readability and reducing bugs.

5. **Performance**: `__syncthreads()` within a warp has minimal overhead, so there's little performance penalty for using it correctly.

## Exercise 6
If a CUDA device's SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM?

**A. 128 threads per block**  
Maximum blocks by thread limit: ⌊1536/128⌋ = 12 blocks  
Actual blocks (limited by SM): min(12, 4) = 4 blocks  
Total threads = 4 × 128 = 512 threads

**B. 256 threads per block**  
Maximum blocks by thread limit: ⌊1536/256⌋ = 6 blocks  
Actual blocks (limited by SM): min(6, 4) = 4 blocks  
Total threads = 4 × 256 = 1024 threads

**C. 512 threads per block**  
Maximum blocks by thread limit: ⌊1536/512⌋ = 3 blocks  
Actual blocks (limited by SM): min(3, 4) = 3 blocks  
Total threads = 3 × 512 = **1536 threads**

**D. 1024 threads per block**  
Maximum blocks by thread limit: ⌊1536/1024⌋ = 1 block  
Actual blocks (limited by SM): min(1, 4) = 1 block  
Total threads = 1 × 1024 = 1024 threads

### Solution
The configuration that results in the most threads is **C (512 threads per block)** with 1536 threads.

## Exercise 7
Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level.

**A. 8 blocks with 128 threads each**  
Total threads = 8 × 128 = 1024 ≤ 2048 ✓  
Number of blocks = 8 ≤ 64 ✓  
**Possible** - Occupancy = 1024/2048 = **50%**

**B. 16 blocks with 64 threads each**  
Total threads = 16 × 64 = 1024 ≤ 2048 ✓  
Number of blocks = 16 ≤ 64 ✓  
**Possible** - Occupancy = 1024/2048 = **50%**

**C. 32 blocks with 32 threads each**  
Total threads = 32 × 32 = 1024 ≤ 2048 ✓  
Number of blocks = 32 ≤ 64 ✓  
**Possible** - Occupancy = 1024/2048 = **50%**

**D. 64 blocks with 32 threads each**  
Total threads = 64 × 32 = 2048 ≤ 2048 ✓  
Number of blocks = 64 ≤ 64 ✓  
**Possible** - Occupancy = 2048/2048 = **100%**

**E. 32 blocks with 64 threads each**  
Total threads = 32 × 64 = 2048 ≤ 2048 ✓  
Number of blocks = 32 ≤ 64 ✓  
**Possible** - Occupancy = 2048/2048 = **100%**

### Solution
**All configurations are possible**. The highest occupancy is achieved by configurations D and E at **100%**.

## Exercise 8
Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64K (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.

**A. The kernel uses 128 threads per block and 30 registers per thread.**  
Max blocks by thread limit: ⌊2048/128⌋ = 16 blocks  
Max blocks by block limit: 32 blocks  
Registers per block: 128 × 30 = 3,840 registers  
Max blocks by register limit: ⌊65,536/3,840⌋ = 17 blocks  
Limiting factor: Thread limit (16 blocks)  
Total threads: 16 × 128 = 2,048  
**Full occupancy achieved: 100%**

**B. The kernel uses 32 threads per block and 29 registers per thread.**  
Max blocks by thread limit: ⌊2048/32⌋ = 64 blocks  
Max blocks by block limit: 32 blocks ← **Limiting factor**  
Registers per block: 32 × 29 = 928 registers  
Max blocks by register limit: ⌊65,536/928⌋ = 70 blocks  
Actual blocks: 32 (limited by block limit)  
Total threads: 32 × 32 = 1,024  
**Occupancy: 1,024/2,048 = 50%**

**C. The kernel uses 256 threads per block and 34 registers per thread.**  
Max blocks by thread limit: ⌊2048/256⌋ = 8 blocks  
Max blocks by block limit: 32 blocks  
Registers per block: 256 × 34 = 8,704 registers  
Max blocks by register limit: ⌊65,536/8,704⌋ = 7 blocks ← **Limiting factor**  
Actual blocks: 7 (limited by registers)  
Total threads: 7 × 256 = 1,792  
**Occupancy: 1,792/2,048 = 87.5%**

## Exercise 9
A student mentions that they were able to multiply two 1024 × 1024 matrices using a matrix multiplication kernel with 32 × 32 thread blocks. The student is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?

### Solution
**This claim is impossible.** A 32 × 32 thread block contains 32 × 32 = 1,024 threads, which exceeds the device's maximum limit of 512 threads per block. CUDA would reject this kernel launch with an error.

To perform 1024 × 1024 matrix multiplication on this device, the student would need to use smaller thread blocks, such as:
- 16 × 16 = 256 threads per block, or  
- 22 × 23 = 506 threads per block (irregular but within limits)

The student likely either misremembered the block dimensions or was using a different device with higher thread limits.