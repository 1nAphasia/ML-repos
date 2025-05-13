# 引入需要的模块
import os
import zipfile
import random
import json
import torch
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

random.seed(200)

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

data_root = "dataset/Chinese Medicine"
dataset = datasets.ImageFolder(data_root, transform=transform)

total_size = len(dataset)
val_size = total_size // 8
train_size = total_size - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)


def VGG_block(in_channels, out_channels, num_convs):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((2, 64), (2, 128), (3, 256), (3, 256), (3, 256))


def vgg(conv_arch):
    net = []
    in_channels = 3
    for num_convs, out_channels in conv_arch:
        net.append(VGG_block(in_channels, out_channels, num_convs))
        in_channels = out_channels
    return nn.Sequential(
        *net,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 5),
    )


net = vgg()


lr, num_epochs, batch_size = 0.1, 10, 8
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters, lr=lr, momentum=0.9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

for epoch in range(num_epochs):
    net.train()
    for batch in train_loader:
        inputs, labels = batch
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

torch.save(net.state_dict(), "vgg_model.pth")

net.eval()
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        print(f"Test Accuracy:{correct/total:.2%}")
