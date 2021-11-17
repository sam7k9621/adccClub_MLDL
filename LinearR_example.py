import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(color_codes=True)

d = 1
n = 50

x_train_tensor = 3*torch.rand(n,d)
y_train_tensor = 3*x_train_tensor+5+0.2*torch.randn(n,d)

plt.scatter(x_train_tensor.numpy(), y_train_tensor.numpy())
plt.xlabel('Volume (x)')
plt.ylabel('Weight (s)')
plt.show()

lr = 1e-1
epochs = 5000
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
for epoch in range(epochs):
    yhat =  a * x_train_tensor + b
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()
    
    a_grad = -2 * (x_train_tensor * error).mean()
    b_grad = -2 * error.mean()
    
    a = a - lr * a_grad
    b = b - lr * b_grad
    
print(a, b)

PDF_x = torch.from_numpy(np.linspace(0,3,1000)).float().view(-1, d)
PDF_y = a.detach().numpy()[0]*PDF_x + b.detach().numpy()[0]

plt.plot(PDF_x.numpy(), PDF_y.numpy(), 'r')
plt.scatter(x_train_tensor.numpy(), y_train_tensor)
plt.xlabel('Volume (x)')
plt.ylabel('Weight (s)')
plt.show()

a = np.arange(2.5, 3.5, 0.025)
b = np.arange(4.5, 5.5, 0.0025)
A, B = np.meshgrid(a, b)
Z = 0
for x, y in zip(x_train_tensor, y_train_tensor):
    yhat =  A * x.item() + B 
    error = y.item() - yhat
    Z += (error ** 2)   

Z /= 50
    
fig = plt.figure()

ax = plt.axes()
CS = ax.contourf(A, B, Z, cmap='RdBu', alpha=0.5)
fig.colorbar(CS)
plt.xlabel('Parameter a')
plt.ylabel('Parameter b')
plt.show()

# 建立 3D 圖形
fig = plt.figure()
ax = fig.gca(projection='3d')

# 繪製 Wireframe 圖形
ax.view_init(60, 100)
CS = ax.plot_surface(A, B, Z, cmap='seismic')
fig.colorbar(CS)
plt.xlabel('Parameter a')
plt.ylabel('Parameter b')
# 顯示圖形
plt.show()
