import torch
import torch.nn.functional as F

# 创建一个测试张量
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# 用 PyTorch 官方的 silu
output = F.silu(x)
print("Input:", x)
print("Output:", output)

# 手动计算 sigmoid
sigmoid_x = torch.sigmoid(x)
print("Sigmoid(x):", sigmoid_x)

# 手动计算 silu
manual_silu = x * sigmoid_x
print("Manual SiLU:", manual_silu)

print("Are they equal?", torch.allclose(output, manual_silu))
