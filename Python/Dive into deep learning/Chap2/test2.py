import torch

x=torch.arange(4.0)


# 告诉pytorch这个变量我要求导
x.requires_grad_(True)
# print(f'(x): {(x.grad)}')

y=2*torch.dot(x,x)

# print(f'y: {y}')

# 反向传播计算x的导数(向量求导)
y.backward()
print(f'x.grad: {x.grad}')

# print(f'x.grad==4*x: {x.grad==4*x}')

#pytorch求导默认导数会累积,所以在重计算的时候需要清零之

x.grad.zero_()
y=x.sum()
y.backward()
print(f'x.grad: {x.grad}')

# 通过y.sum将y作为标量,使得y可以作为标量被求导,这是机器学习最常见的事情
x.grad.zero_()
y=x*x
y.sum().backward()
print(f'x.grad: {x.grad}')

#
x.grad.zero_()
y=x*x
u=y.detach()
z=u*x

z.sum().backward()
# print(f'x.grad==u: {x.grad==u}')

def f(a):
    b=a*2
    while b.norm()<1000:
        b=b*2
    if b.sum()>0:
        c=b
    else:
        c=100*b
    return c

a=torch.randn(size=(),requires_grad=True)
d=f(a)
d.backward()