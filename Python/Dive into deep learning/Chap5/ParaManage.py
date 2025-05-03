import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(f"net(X):{net(X)}")

print(net[2].state_dict())

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(f"net[2].weight.grad == None:{net[2].weight.grad == None}")
