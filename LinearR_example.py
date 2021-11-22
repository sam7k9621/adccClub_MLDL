import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(color_codes=True)

# Seed
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

d = 1
n = 50

x_trn_tensor = 3*torch.rand(n,d)
s_trn_tensor = 3*x_trn_tensor+5+0.2*torch.randn(n,d)

plt.scatter(x_trn_tensor.numpy(), s_trn_tensor.numpy())
plt.xlabel('Volume (x)')
plt.ylabel('Weight (s)')
plt.show()


lr = 1e-1
epochs = 5000
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
for epoch in range(epochs):
    yhat =  a * x_trn_tensor + b
    error = s_trn_tensor - yhat
    loss = (error ** 2).mean()
    
    a_grad = -2 * (x_trn_tensor * error).mean()
    b_grad = -2 * error.mean()
    
    a = a - lr * a_grad
    b = b - lr * b_grad
    
PDF_x = torch.from_numpy(np.linspace(0,3,1000)).float().view(-1, d)
PDF_y = a.detach().numpy()[0]*PDF_x + b.detach().numpy()[0]

plt.plot(PDF_x.numpy(), PDF_y.numpy(), 'r')
plt.scatter(x_trn_tensor.numpy(), s_trn_tensor)
plt.xlabel('Volume (x)')
plt.ylabel('Weight (s)')
plt.show()

