import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

X = torch.randn(200,1)*10
y = X+3*torch.randn(200,1)

class LinearRegressionModel(torch.nn.Module):

  def __init__(self):
    super(LinearRegressionModel,self).__init__()
    self.linear = torch.nn.Linear(1,1)

  def forward(self,x):
    pred = self.linear(x)
    return pred

model = LinearRegressionModel()

# 모델 파라미터 초기화
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

print(model)
print(list(model.parameters()))

w,b = model.parameters()
w1,b1 = w[0][0].item(),b[0].item()
x1 = np.array([-30,30])
y1 = w1 * x1+b1

plt.plot(x1,y1,'r')
plt.scatter(X,y)
plt.grid()
plt.show()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

epochs = 100
losses = []
for epoch in range(epochs):
  optimizer.zero_grad()
  y_pred = model(X)
  loss = criterion(y_pred,y)
  losses.append(loss.item())
  loss.backward()

  optimizer.step()
  print(f'epoch: {epoch}, loss: {loss.item()}')

plt.plot(range(epochs),losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

w1,b1 = w[0][0].item(),b[0].item()
x1 = np.array([-30,30])
y1 = w1 * x1+b1

plt.plot(x1,y1,'r')
plt.scatter(X,y)
plt.grid()
plt.show()