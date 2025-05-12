import torch
import torch.nn as nn
import time

m, n = 1024, 1024
input = torch.arange(1, m*n+1).reshape(m,n).float()

layer_norm = nn.LayerNorm(n, elementwise_affine=False, eps=1e-6).cuda()

for i in range(10):
    output = layer_norm(input.cuda())

# measure time
start = time.time()
for i in range(1000):
    output = layer_norm(input.cuda())
torch.cuda.synchronize()
end = time.time()

pytorch_time = (end - start)/1000
print(f"PyTorch LayerNorm time: {pytorch_time * 1000:.4f} ms")