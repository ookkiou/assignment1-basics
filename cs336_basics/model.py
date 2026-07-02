import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # 初始化权重和偏置
        # weight 形状: [d_out, d_in]
        # bias 形状: [d_out]
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * 0.01)
        self.bias = nn.Parameter(torch.zeros(d_out))
    
    def forward(self, x):
        # x 形状: [..., d_in]
        # 输出形状: [..., d_out]
        return x @ self.weight.T + self.bias


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.01)
    
    def forward(self, token_ids):
        return self.weight[token_ids]
