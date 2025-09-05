## Exercise 1
In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.

**A. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design.**
### A. Solution
```cpp
01 __global__ void matrixmulti1rowforthread(float* A, float* B, float* C, int M, int K, int N){
02    int row = threadIdx.y + blockDim.y * blockIdx.y;
03    if (row < M){
04        for(int col =0 ; col < N ; col++){
05            float val = 0.0;
06            for (int i = 0; i < K ; i++){
07                val += A[i + row * K] * B[col + i * N];
08            }
09            C[col + row * N] = val;
10        }
11    }
12}
```

**B. Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design.**
### B. Solution
```cpp
01 __global__ void matrixmulti1colforthread(float* A, float* B, float* C, int M, int K, int N){
02    int col = threadIdx.x + blockDim.x * blockIdx.x;

03    if (col < N){
04        for(int row = 0 ; row < M ; row++){
05            float val = 0.0;
06            for (int i = 0; i < K ; i++){
07                val += A[row * K + i] * B[col + i * N];
08            }
09            C[col + row * N] = val;
10        }
11    }
12}
```

## Exercise 2
Amatrix-vectormultiplicationtakesaninputmatrixBandavectorCand produces one output vector A. Each element of the output vector A is the dot
product of one row of the input matrix B and C, that is, **A[i] = sum_over_j(B[i][j] * C[j])**.
For simplicity we will handle only square matrices whose elements are single- precision floating-point numbers. Write a matrix-vector multiplication kernel and the host stub function that can be called with four parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. Use one thread to calculate an output vector element.

### Solution
```cpp
__global__ void matrix_vector_mult(float* B, float* C, float* A, int M, int N){
    int i = threadIdx.x + blockDim.x *    blockIdx.x;

    if (i < M){
        float val = 0.0;
        for (int j = 0 ; j < N ; j ++){
            val += B[i * N + j] * C[j];
        }
        A[i] = val;
    }
}
```

## Exercise 3
Consider the following CUDA kernel and the corresponding host function that calls it:
```cpp
01 __global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
02     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
03     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
04     if (row < M && col < N) {
05         b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
06     }
07 }
08 void foo(float* a_d, float* b_d) {
09     unsigned int M = 150;
10     unsigned int N = 300;
11     dim3 bd(16, 32);
12     dim3 gd((N - 1) / 16 + 1, ((M - 1) / 32 + 1));
13     foo_kernel <<< gd, bd >>> (a_d, b_d, M, N);
14 }
```

**A. What is the number of threads per block?** <br>
**(512)** threads per block (16 threads per block in x direction and 32 threads per block in y direction)

**B. What is the number of threads in the grid?** <br>
**(48,640)** threads in the grid (There are 512 threads per block and we have 95 blocks so (95 * 512) = 48,640 threads)

**C. What is the number of blocks in the grid?** <br>
**(95)** blocks in the grid (19 blocks in x direction and 5 blocks in y direction so (19 * 5) = 95 blocks)

**D. What is the number of threads that execute the code on line 05?** <br>
**(48,640)**

## Exercise 4

Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10:

**A. If the matrix is stored in row-major order.** <br>
**8010** The row-major order is caluculated as follows: (row * width) + col = ((20 * 400)) + 10 = 8010

**B. If the matrix is stored in column-major order.** <br>
**5020** The column-major order is caluculated as follows: (col * height) + row = ((10 * 500)) + 20 = 5020


## Exercise 5
Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x=5, y=20, and z=5.

**1008010** It calculated as (plane x width x height + row x width + col) =  ((5 * 400 * 500) + 20 * 400 + 10)
