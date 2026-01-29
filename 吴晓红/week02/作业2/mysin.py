import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)  # 形状: (200, 1)
y_numpy = np.sin(x_numpy) + np.random.randn(200, 1)
x = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()
print("数据生成完成。")
print("---" * 10)
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = SimpleClassifier(1, 10, 1)
a = torch.randn(1,requires_grad=True, dtype = torch.float)
b = torch.randn(1, requires_grad=True, dtype = torch.float)
print(f"初始参数 a: {a.item():.4f}")
print(f"初始参数 b: {b.item():.4f}")
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
num_epochs = 10000
for epoch in range(num_epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
a_learned = a.item()
b_learned = b.item()
print(f"拟合的斜率 a: {a_learned:.4f}")
print(f"拟合的截距 b: {b_learned:.4f}")

with torch.no_grad():
    y_predicted = model(x).numpy()  # 转换为numpy数组（方便绘图）

# 绘制原始数据、真实sin函数和拟合曲线
plt.figure(figsize=(12, 8))
plt.scatter(x_numpy, y_numpy, label='Raw data (sin + noise)', color='blue', alpha=0.6)
plt.plot(x_numpy, np.sin(x_numpy), label='True sin function', color='green', linewidth=3)
plt.plot(x_numpy, y_predicted, label='Fitted model (MLP)', color='red', linewidth=3)
plt.xlabel('X (radians)')
plt.ylabel('y')
plt.title('Fitting Sin Function with Multi-layer Perceptron')
plt.legend()
plt.grid(True)
plt.show()