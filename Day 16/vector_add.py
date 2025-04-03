import time
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
        a_ptr,
        b_prt,
        c_ptr,
        no_elments: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < no_elments
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_prt + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def vector_add_triton(a: torch.Tensor, b: torch.Tensor, BLOCK_SIZE:int = 1024) -> torch.Tensor:
    assert a.numel() == b.numel(), "Input vectors must have the same number of elements."
    no_elments = a.numel()
    c = torch.empty_like(a)

    grid = lambda meta: (triton.cdiv(no_elments, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](
        a,
        b,
        c,
        no_elments,
        BLOCK_SIZE
    )
    return c

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

if __name__ == "__main__":
    n = 328483
    a = torch.arange(0, n, device='cuda', dtype=torch.float32)
    b = torch.arange(n, 2 * n, device='cuda', dtype=torch.float32)

    c_triton = vector_add_triton(a, b)
    c_torch = a + b

    def torch_add(a, b):
        return a + b

    if torch.allclose(c_triton, c_torch):
        print("Success: Triton kernel result matches PyTorch result!")
    else:
        print("Error: The results do not match.")

    triton_time = benchmark(vector_add_triton, a, b)
    print(f"Triton time: {triton_time:.2f} ms")

    torch_time = benchmark(torch_add, a, b)
    print(f"PyTorch time: {torch_time:.2f} ms")