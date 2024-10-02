import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 创建模型、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 初始化 TensorBoard SummaryWriter
writer = SummaryWriter()

# 生成一些简单的训练数据
x_train = torch.randn(100, 10)  # 100 个样本，每个样本 10 维
y_train = torch.randn(100, 1)   # 100 个目标值

# 训练模型，并将损失记录到 TensorBoard
for epoch in range(5):  # 训练 5 个 epoch
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 每 10 个 epoch 记录一次损失
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')
        writer.add_scalar('training_loss', loss.item(), epoch)

# 关闭 TensorBoard 记录器
writer.close()
