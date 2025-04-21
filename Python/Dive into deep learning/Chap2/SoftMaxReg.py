import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True
)

mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans
)


def get_fashion_mnist_labels(lebels):
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[int(i)] for i in lebels]


def show_images(img, num_rows, num_cols, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, img)):
        ax.imshow(img.numpy())
    else:
        ax.imshow(img)


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18, shuffle=True)))

show_images(X.reshape(18, 1, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

batch_size = 256


def get_dataloader_workers():
    return 4


train_iter = data.DataLoader(
    mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()
)
timer = d2l.Timer()
for X, t in train_iter:
    continue
print(f"timer.stop():.2f:{timer.stop():.2f} sec")
