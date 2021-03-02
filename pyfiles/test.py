import numpy as np
import torch

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])

W = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W], lr=0.15)

n_epochs = 10

for epoch in range(n_epochs + 1):
    h = x_train * W

    cost = torch.mean((h-y_train) ** 2)

    print('Epoch {:4d}/{} W: {:.3f} Cost: {:.6f}'.format(epoch, n_epochs, W.item(), cost.item()))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()