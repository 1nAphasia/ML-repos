import torch

# 基础：顺序生成arange(),形状.shape(),元素总数.numel(),改变形状.reshape()
# x=torch.arange(12)
#print(f'x:{x}')

# print(f'x.numel:{x.numel()}')

# X=x.reshape(3,4)
# X=x.reshape(3,-1) #-1会自动补全
# X=x.reshape(-1,4)

# print(f'X:{X}')
# 生成全0,全1,标准高斯分布采样随机数,有初始值的张量
#print(f'torch.zeros((2,3,4)):{torch.zeros((2,3,4))}')

# print(f'torch.randn(3,4):{torch.randn(3,4)}')
# 运算符：+ - * / **,以及指定函数后自动进行的逐元素运算

# x=torch.tensor([1.0,2,4,8])
# y=torch.tensor([2,2,2,2])
# print(f'x+y,x-y,x*y,x/y,x**y:{x+y,x-y,x*y,x/y,x**y}')
# print(f'torch.exp(x):{torch.exp(x)}')

# 张量链接cat(*,dim=?) 

# X=torch.arange(12,dtype=torch.float32).reshape((3,4))
# Y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
# print(f'torch.cat((X,Y),dim=0),torch.cat((X,Y),dim=1):{torch.cat((X,Y),dim=0),torch.cat((X,Y),dim=1)}')

# 根据逻辑运算符构建二元张量

# print(f'X==Y:{X==Y}')
# print(f'X.sum():{X.sum()}')

# 广播机制：通过适当复制元素来扩展一个或两个数组,从而使转换后,两个向量具有相同形状。之后对生成的数组进行逐元素操作。

# a=torch.arange(3).reshape((3,1))
# b=torch.arange(2).reshape((1,2))
# print(f'a,b:{a,b}')
# print(f'a+b:{a+b}')

# 索引和切片：可以用[-1]选择最后一个元素,用[1:3]选择第二和第三个元素。还能指定索引将元素写入矩阵,还能通过索引批量指定位置来将指定元素写入矩阵：

# X=torch.arange(12,dtype=torch.float32).reshape((3,4))
# print(f'X[-1],X[1:3]:{X[-1],X[1:3]}')

# X[1,2]=9
# print(f'X:{X}')

# X[0:2,:]=12
# print(f'X:{X}')

# 节省内存:执行Y+X的操作会解除对Y的引用,为Y重新分配内存,这样可能是不好的。切片表示法可以保证变量被原地赋值,而不被解除引用

X=torch.arange(12,dtype=torch.float32).reshape((3,4))
Y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])

before=id(Y)
Y=Y+X
id(Y)==before

Z=torch.zeros_like(Y)
print(f'id(Z):{id(Z)}')
Z[:]=X+Y
print(f'id(Z):{id(Z)}')

# 如果之后X不再使用,也可以通过切片法或者+=这种运算符来减少内存开销

before=id(X)
X+=Y
print(f'id(X)==before:{id(X)==before}')

# 转化为其他python对象：tensor与ndarray共享底层内存,转换为原生python类型也很方便。

X=torch.arange(12,dtype=torch.float32).reshape((3,4))

A=X.numpy()
B=torch.tensor(A)
print(f'type(A),type(B):{type(A),type(B)}')

a=torch.tensor([3.5])

print(f'a,a.item(),float(a),int(a):{a,a.item(),float(a),int(a)}')