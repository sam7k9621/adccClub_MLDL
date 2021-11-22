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

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.func = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        return self.func(x)


lr = 1e-1
epochs = 5000

model=NeuralNetwork()

MSELoss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
for epoch in range(epochs):
    model.train()
    yhat = model(x_trn_tensor)
    
    loss = MSELoss(s_trn_tensor, yhat)
    loss.backward()    
    optimizer.step()
    optimizer.zero_grad()

PDF_x = torch.from_numpy(np.linspace(0,3,1000)).float().view(-1, d)
PDF_y = model(PDF_x)
plt.scatter(x_trn_tensor.numpy(), s_trn_tensor)
plt.plot(PDF_x.detach().numpy(), PDF_y.detach().numpy(), 'r')
plt.xlabel('Volume (x)')
plt.ylabel('Weight (s)')
plt.show()
