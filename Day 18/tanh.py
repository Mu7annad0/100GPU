@triton.jit
def tanh_kernel(
    A_ptr, C_ptr,
    M, N,
    stride_am, stride_an,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(axis=0)  # row
    pid_n = tl.program_id(axis=1)  # column
    
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    
    # Initialize pointers for each row
    A_ptrs = A_ptr + m_offsets[:, None] * stride_am + n_offsets[None, :] * stride_an
    
    # Load values with mask
    mask = m_mask[:, None] & n_mask[None, :]
    x = tl.load(A_ptrs, mask=mask)
    
    # Compute tanh: (exp(2x) - 1) / (exp(2x) + 1)
    exp_2x = tl.exp(2 * x)
    result = (exp_2x - 1) / (exp_2x + 1)
    
    # Store the result
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, result, mask=mask)

def tanh(x: torch.Tensor, BLOCK_SIZE_M: int = 32, BLOCK_SIZE_N: int = 32) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    stride_am, stride_an = x.stride()
    stride_cm, stride_cn = y.stride()

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    tanh_kernel[grid](
        x, y,
        M, N,
        stride_am, stride_an,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return y

def benchmark(func, *args, n_warmup=10, n_iters=100):
    """
    Benchmarks a function by performing warm-up iterations followed by timed iterations.

    Args:
        func (callable): The function to benchmark.
        *args: Arguments to pass to the function.
        n_warmup (int): Number of warm-up iterations.
        n_iters (int): Number of iterations for timing.

    Returns:
        float: Average execution time per iteration in milliseconds.
    """
    # Warm-up: execute the function several times to mitigate initial overhead.
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()  # Wait for all GPU operations to finish.

    # Timing the execution.
    start = time.perf_counter()
    for _ in range(n_iters):
        func(*args)
    torch.cuda.synchronize()  # Ensure all GPU operations are complete.
    end = time.perf_counter()

    avg_time_ms = (end - start) / n_iters * 1000
    return avg_time_ms


if __name__ == '__main__':
    M, N = 1024*2, 1024
    x = torch.randn((M, N), device='cuda', dtype=torch.float32)
    
    y_triton = tanh(x)
    y_torch = torch.tanh(x)

    if torch.allclose(y_triton, y_torch, atol=1e-5, rtol=1e-5):
        print("Success: Triton kernel result matches PyTorch result!")
    else:
        print("Error: The results do not match.")

    triton_time = benchmark(tanh, x)
    print(f"Triton time: {triton_time:.2f} ms")

    torch_time = benchmark(torch.tanh, x)
    print(f"PyTorch time: {torch_time:.2f} ms")
    