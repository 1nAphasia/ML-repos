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


def show_images(img, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, img)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()
    return axes


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18, shuffle=True)))

show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

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


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(
            root="../data", train=True, transform=trans, download=True
        )
        mnist_test = torchvision.datasets.FashionMNIST(
            root="../data", train=False, transform=trans, download=True
        )
    return (
        data.DataLoader(
            mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()
        ),
        data.DataLoader(
            mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()
        ),
    )


train_iter, test_iter = load_data_fashion_mnist(32, resize=96)
for X, y in train_iter:
    print(X.shape, X.dtype)
    print(y.shape, y.dtype)
    break
