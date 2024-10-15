"""
    @time:2024/10/10 16:38
    @file:17_right_rate.py
    分类问题的正确率
"""

import torch

outputs = torch.tensor([[0.1, 0.2],
                        [0.3, 0.4]])
# argmax参数1代表横向比较，0.1和0.2比较，0.2大，位置取1；参数0代表纵向比较，0.1和0.3比较
# 实际分类
preds = outputs.argmax(1)
print(preds)
# 正确分类
targets = torch.tensor([0, 1])

print(preds == targets)
# 计算对应位置正确的个数
print((preds == targets).sum())
