import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 데이터 생성
X = torch.randn(1000, 1) * 10
y = X + 3 * torch.randn(1000, 1)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        pred = self.linear(x)
        return pred

model = LinearRegressionModel()

# 모델 파라미터 초기화
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 100
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    loss.backward()

    optimizer.step()
    print(f'epoch: {epoch}, loss: {loss.item()}')

# 모델 평가
model.eval()  # 평가 모드로 설정
with torch.no_grad():  # 기울기 계산 비활성화
    y_pred_test = model(X_test)
    mse = mean_squared_error(y_test.numpy(), y_pred_test.numpy())
    r2 = r2_score(y_test.numpy(), y_pred_test.numpy())

print(f'Test MSE: {mse}')
print(f'Test R²: {r2}')

# 그래프 표시 (학습 데이터 및 테스트 데이터)
w, b = model.parameters()
w1, b1 = w[0][0].item(), b[0].item()
x1 = np.array([-30, 30])
y1 = w1 * x1 + b1

plt.plot(x1, y1, 'r')
plt.scatter(X_train.numpy(), y_train.numpy(), label='Train Data')
plt.scatter(X_test.numpy(), y_test.numpy(), label='Test Data')
plt.grid()
plt.legend()
plt.show()

plt.plot(range(epochs), losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()