import time
import torch
import triton
import triton.language as tl

@triton.jit
def relu_forward_kernel(
        x_ptr,
        y_ptr,
        no_elments: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < no_elments
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(0.0, x)
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.jit
def relu_backward_kernel(
        x_ptr,
        output_grad_ptr,
        input_grad_ptr,
        no_elments: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < no_elments
    x = tl.load(x_ptr + offsets, mask=mask)
    grad_out = tl.load(output_grad_ptr + offsets, mask=mask)
    grad_in = tl.where(x > 0, grad_out, 0.0)
    tl.store(input_grad_ptr + offsets, grad_in, mask=mask)

class TritonReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x:torch.Tensor, BLOCK_SIZE:int = 1024) -> torch.Tensor:
        no_elments = x.numel()
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(no_elments, meta['BLOCK_SIZE']),)
        relu_forward_kernel[grid](x, y, no_elments, BLOCK_SIZE)
        ctx.save_for_backward(x)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple:
        x, = ctx.saved_tensors
        no_elments = x.numel()
        input_grad = torch.zeros_like(x)
        grid = lambda meta: (triton.cdiv(no_elments, meta['BLOCK_SIZE']),)
        relu_backward_kernel[grid](x, grad_out, input_grad, no_elments, ctx.BLOCK_SIZE)
        return input_grad, None

def triton_relu(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    return TritonReLUFunction.apply(x, BLOCK_SIZE)

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

    # Create a random input tensor on the GPU with gradient tracking.
    N = 283893
    torch.cuda.empty_cache()
    x = torch.randn(N, device='cuda', dtype=torch.float32, requires_grad=True)
    BLOCK_SIZE = 1024

    # Forward pass using our custom Triton ReLU.
    y_triton = triton_relu(x, BLOCK_SIZE)
    # Define a dummy loss (sum of outputs) and perform backward pass.
    loss_triton = y_triton.sum()
    loss_triton.backward()
    
    # For validation, compare against PyTorch's built-in ReLU.
    x_torch = x.detach().clone().requires_grad_()
    y_torch = torch.relu(x_torch)
    loss_torch = y_torch.sum()
    loss_torch.backward()

    # Check if the gradients match.
    if torch.allclose(x.grad, x_torch.grad, atol=1e-4):
        print("Success: Triton autograd ReLU backward matches PyTorch!")
    else:
        print("Error: The gradients do not match.")

    # Benchmark the forward pass.
    triton_time = benchmark(lambda: triton_relu(x, BLOCK_SIZE))
    torch_time = benchmark(lambda: torch.relu(x))
    print(f"Average execution time (Forward Pass):")
    print(f"  Triton ReLU = {triton_time:.3f} ms")
    print(f"  PyTorch ReLU = {torch_time:.3f} ms")