import torch

x_data = torch.tensor([[5.0, 1.5],[4.8, 1.2],[5.1, 1.9],[4.9, 1.8],[6.1, 3.8],[6.3, 3.9],[6.8, 4.1],[6.4, 4.2]],dtype=torch.float)
y_data = torch.tensor([[1,1,1,1,0,0,0,0]],dtype=torch.float).t()

class LogisticRegression(torch.nn.Module):
     def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(2, 1)
     def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5000):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(x_data)
    # Compute Loss
    loss = criterion(y_pred, y_data)
    # Backward pass
    loss.backward()
    optimizer.step()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
