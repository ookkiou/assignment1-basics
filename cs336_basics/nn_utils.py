import torch,math


def silu(x): #激活函数
    return x * torch.sigmoid(x)


def softmax(x, dim): #出现词的概率
    # 1. 先求 x 在 dim 上的最大值
    # 2. x 减去最大值
    # 3. 求指数
    # 4. 除以指数的和
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum


def cross_entropy(logits, labels): #交叉熵损失：模型预测的概率分布与真实标签的概率分布之间的差异
    # 直接用 LogSoftmax 代替 Softmax + Log，避免 e^x 溢出
    log_probs = torch.log_softmax(logits, dim=1) 
    batch_indices = torch.arange(len(labels))
    loss = -log_probs[batch_indices, labels].mean()
    return loss


def gradient_clipping(parameters, max_l2_norm):
    # 1. 收集所有需要梯度的参数的 grad
    # 2. 计算这些 grad 的总 L2 范数
    # 3. 如果范数超过 max_l2_norm，按比例缩放所有 grad
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    # 1. 如果在 warmup 阶段
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    # 2. 如果在 cosine 衰减阶段（注意这里！）
    elif it < cosine_cycle_iters:  # 不是 warmup_iters + cosine_cycle_iters
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)  # 这里也要改
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    # 3. 如果超出了，返回 min_lr
    else:
        return min_learning_rate
