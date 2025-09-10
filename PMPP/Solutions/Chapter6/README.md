## Exercise 1
Write a matrix multiplication kernel function that corresponds to the design illustrated in **Fig. 6.4**.

### Solution
```c
__global__ void tiled_matmul(float *A, float *B, float *C, int M, int K, int N){
    __shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float val = 0.0;

    for (int phase = 0; phase < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++phase){
        // ---------------- Load A tile into shared memory ----------------
        // Each thread loads one element of A from global memory

        int a_col = phase*TILE_WIDTH + tx;
        if (row < M && a_col < K) {
            // Access is row-major: consecutive threads load consecutive elements
            A_shared[ty][tx] = A[row*K + a_col];
        } else {
            A_shared[ty][tx] = 0.0f;
        }

        /// ---------------- Load B tile into shared memory ----------------
        // Each thread loads one element of B, but we TRANSPOSE it when storing
        // This ensures global memory accesses are coalesced
        int b_row = phase*TILE_WIDTH + ty;
        if (b_row < K && col < N) {
            // Store transposed: (tx, ty) instead of (ty, tx)
            B_shared[tx][ty] = B[b_row*N + col];
        } else {
            B_shared[tx][ty] = 0.0f;
        }

        __syncthreads();

        // ---------------- Compute partial results ----------------
        // Multiply the row of A_shared by the column of B_shared
        // Note: we use the transposed version of B_shared here
        for (int j = 0; j < TILE_WIDTH; ++j) {
            val += A_shared[ty][j] * B_shared[tx][j];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row*N + col] = val;
    }
}
```

## Exercise 2
For tiled matrix multiplication, of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will the kernel completely avoid uncoalesced accesses to global memory? (You need to consider only square blocks.)

### Solution
A GPU executes threads in groups called warps (typically 32 threads). When threads in a warp access global memory, the hardware tries to combine their individual requests into a single, large transaction. This is called a coalesced memory access and is highly efficient. It occurs when the 32 threads access consecutive memory addresses. If they access scattered addresses, the hardware must issue multiple, inefficient transactions, which is called uncoalesced access and severely degrades performance.

### 1. Access Pattern for Matrix A
In a standard tiled implementation, each thread (tx, ty) in a BLOCK_SIZE x BLOCK_SIZE thread block loads an element of matrix A. The global memory address accessed is typically indexed like A[row][... + tx].

- Threads within a warp have consecutive thread IDs.
- When BLOCK_SIZE is a multiple of the warp size (e.g., 32), the first warp consists of threads where ty=0 and tx ranges from 0 to 31.
- These threads access A[some_row][col], A[some_row][col+1], ..., A[some_row][col+31].
- Since the matrix is stored in row-major order, these elements are perfectly contiguous in memory.

This results in a fully coalesced access for Matrix A, regardless of the BLOCK_SIZE.

### 2. Access Pattern for Matrix B
The challenge lies in accessing Matrix B. A thread (tx, ty) typically loads an element from B indexed like B[... + ty][... + tx]. The key is to understand how the 2D thread block (tx, ty) maps to a 1D warp. The mapping is done in a row-by-row fashion: the linear thread ID is id = ty * BLOCK_SIZE + tx. A warp consists of 32 threads with consecutive linear IDs.

**The Problem Case**: BLOCK_SIZE **is NOT a multiple of warp size**
Let's assume warpSize = 32 and we choose a BLOCK_SIZE = 16.

- The first warp consists of threads with linear IDs 0 through 31.
- Threads 0-15 map to (tx=0..15, ty=0). They will access 16 consecutive elements in a row of Matrix B.
- Threads 16-31 map to (tx=0..15, ty=1). They will access another 16 consecutive elements, but from the next row of Matrix B.

Because this single warp requests data from two separate, non-contiguous memory regions (two different rows), the access is uncoalesced. The hardware must issue at least two memory transactions to satisfy the request of this one warp.

**The Solution Case**: BLOCK_SIZE **is a multiple of warp size**


## Exercise 3
Consider the following CUDA kernel:
```cpp
01  **global** void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
02  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
03  **shared** float a_s[256];
04  **shared** float bc_s[4*256];
05  a_s[threadIdx.x] = a[i];
06  for(unsigned int j = 0; j < 4; ++j) {
07      bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i*4 + j];
08  }
09  __syncthreads();
10 d[i + 8] = a_s[threadIdx.x];
11 e[i*8] = bc_s[threadIdx.x*4];
12 }
```
For each of the following memory accesses, specify whether they are coalesced or uncoalesced or coalescing is not applicable:
### A. The access to array a of line 05
The access to array a of line 05 is coalesced.

### B. The access to array a_s of line 05
Shared Memory does not require coalescing.

### C. The access to array b of line 07
The access to array b of line 07 is coalesced.

### D. The access to array c of line 07
The access to array c of line 07 is uncoalesced.

### E. The access to array bc_s of line 07
Shared Memory does not require coalescing.

### F. The access to array a_s of line 10
Shared Memory does not require coalescing.

### G. The access to array d of line 10
The access to array d of line 10 is coalesced

### H. The access to array bc_s of line 11
Shared Memory does not require coalescing.

### I. The access to array e of line 11
The access to array c of line 07 is uncoalesced.
