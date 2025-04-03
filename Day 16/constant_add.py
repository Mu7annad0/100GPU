import torch
import triton
import triton.language as tl

@triton.jit
def constant_add_kernel(
        x_ptr,
        constant,
        y_ptr,
        N0: tl.constexpr,
        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N0
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x + constant
    tl.store(y_ptr + offsets, y, mask=mask)

def constant_add_triton(x: torch.Tensor, constant: float) -> torch.Tensor:
    N0 = x.numel()
    BLOCK_SIZE = N0
    y = torch.empty_like(x)
    gird = lambda meta: (1, )
    constant_add_kernel[gird](x, constant, y, N0, BLOCK_SIZE)
    return y


if __name__ == '__main__':
    N0 = 1024
    x = torch.arange(0, N0, device='cuda', dtype=torch.float32)
    constant = 3.0
    y_triton = constant_add_triton(x, constant)

    y_torch = x + constant

    # Verify correctness.
    if torch.allclose(y_triton, y_torch):
        print("Success: Triton kernel result matches PyTorch result!")
    else:
        print("Error: The results do not match.")