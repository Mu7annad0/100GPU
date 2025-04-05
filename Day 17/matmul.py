import torch
import triton
import triton.language as tl
import time


# Triton kernel for matrix multiplication (no tiling)
@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to the first matrix (2D tensor)
    b_ptr,  # Pointer to the second matrix (2D tensor)
    c_ptr,  # Pointer to the output matrix (2D tensor)
    M,
    N,
    K,  # Matrix dimensions (MxN * NxK = MxK)
    stride_am,
    stride_an,  # Strides for matrix A
    stride_bn,
    stride_bk,  # Strides for matrix B
    stride_cm,
    stride_ck,  # Strides for matrix C
):
    # Thread ID (1D launch grid)
    pid = tl.program_id(axis=0)

    # Compute row and column indices
    row_idx = pid // K
    col_idx = pid % K

    # Check if the thread is within bounds
    if row_idx < M and col_idx < K:
        # Initialize the accumulator
        acc = 0.0

        # Iterate over the N dimension
        for n in range(N):
            # Load elements from A and B
            a_val = tl.load(a_ptr + row_idx * stride_am + n * stride_an)
            b_val = tl.load(b_ptr + n * stride_bn + col_idx * stride_bk)

            # Accumulate the dot product
            acc += a_val * b_val

        # Store the result back to global memory
        tl.store(c_ptr + row_idx * stride_cm + col_idx * stride_ck, acc)


# Triton matrix multiplication wrapper
def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0], "Matrix dimensions must match for multiplication"

    M, N = a.shape
    N, K = b.shape
    c = torch.empty(M, K, device="cuda")

    # Configure the kernel
    grid = lambda meta: (M * K,)  # 1D launch grid (one thread per output element)

    # Launch kernel
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),  # Strides for A
        b.stride(0),
        b.stride(1),  # Strides for B
        c.stride(0),
        c.stride(1),  # Strides for C
    )

    return c


# Timing function
def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


# Test the Triton kernel
if __name__ == "__main__":
    # Generate test data
    M, N, K = 2 * 1024, 2 * 1024, 2 * 1024  # Dimensions for test matrices
    a = torch.rand(M, N, device="cuda")
    b = torch.rand(N, K, device="cuda")

    # Measure Triton kernel time
    output_triton, triton_time = measure_time(matmul_triton, a, b)
    print(f"Triton kernel time: {triton_time * 1000:.4f} ms")

    # Measure PyTorch (GPU) matrix multiplication time
    output_torch_gpu, torch_gpu_time = measure_time(lambda: a @ b)
    print(f"PyTorch (GPU) matrix multiplication time: {torch_gpu_time * 1000:.4f} ms")

    # Validate results
    assert torch.allclose(output_triton, output_torch_gpu, atol=1e-5)
    print("Validation passed!")