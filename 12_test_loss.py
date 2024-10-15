import torch
from requests.packages import target
from torch import nn

nn_inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)


nn_inputs = torch.reshape(nn_inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss= nn.L1Loss()
result1 = loss(nn_inputs, targets)
print(result1)


loss_sum = nn.L1Loss(reduction="sum")
result2 = loss_sum(nn_inputs, targets)
print(result2)
# 均方差损失
loss_mse = nn.MSELoss()
result_mse = loss_mse(nn_inputs, targets)
print(result_mse)

# 交叉熵损失,分类

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)