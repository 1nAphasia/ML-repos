import torch
from d2l import torch as d2l
import torch.nn as nn


def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003

    trainer = torch.optim.SGD(
        [{"param": net[0].weight, "weight_decay": wd}, {"param": net[0].bias}], lr=lr
    )
    animator = d2l.Animator(
        xlabel="epochs",
        ylabel="loss",
        yscale="log",
        xlim=[5, num_epochs],
        legend=["train", "test"],
    )

    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
            if (epoch + 1) % 5 == 0:
                animator.add(
                    epoch + 1,
                    (
                        d2l.evaluate_loss(net, train_iter, loss),
                        d2l.evaluate_loss(net, test_iter, loss),
                    ),
                )
    print("w的L2范数：", net[0].weight.norm().item())
