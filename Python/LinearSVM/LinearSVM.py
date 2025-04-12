import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


class SVM:
    def __init__(self, C=1.0, kernel='linear', tol=1e-3, max_iter=1000):
        self.C = C                      # 正则化参数
        self.kernel = kernel            # 核函数（如'rbf'）
        self.tol = tol                  # 收敛容忍度
        self.max_iter = max_iter        # 最大迭代次数
        self.alpha = None               # 拉格朗日乘子
        self.b = 0                      # 偏置项
        self.X = None                   # 训练数据
        self.y = None                   # 标签（±1）
        self.errors = None              # 误差缓存
        
    def fit(self,X,Y):
        self.X=X
        self.Y=Y
        m=X.shape[0]
        self.alpha=np.zeros(m)
        self.b=0
        self.errors=np.zeros(m)-Y
        num_changed=0
        examine_all=True
        iter_=0
        while(num_changed or examine_all) and iter_<self.max_iter:
            num_changed=0
            if examine_all:
                for i in range(m):
                    num_changed+=self.examine_example(i)
            else:
                non_bound=np.where((self.alpha>0)&(self.alpha<self.C))[0]
                for i in non_bound:
                    num_changed+=self.examine_example(i)
            examine_all=not examine_all if num_changed==0 else True
            iter_+=1
    
    def examine_example(self,i):
        yi=self.y[i]
        Ei=self.errors[i]
        ri=Ei-yi
        if(ri<-self.tol and self.alpha[i]<self.C) or (ri>self.tol and self.alpha>0):
            j=self.select_j(i,Ei)
            if j is None:
                return 0
            if self.optimize(i,j):
                return 1
            else:  
                for j in np.random.permutation(len(self.alpha)):
                    if j!=i and self.optimize(i,j):
                        return 1
            return 0
    
    def select_j(self,i,Ei):
        max_diff=0
        j=-1
        non_bound=np.where((self.alpha>0)&(self.alpha<self.C))[0]
        
        candidates=non_bound if len(non_bound)>1 else range(len(self.alpha))
        for k in candidates:
            if k==i:
                continue
            Ek=self.errors[k]
            diff=abs(Ei-Ek)
            if(diff>max_diff):
                max_diff=diff
                j=k
        return j if j!=-1 else None
            
    
    def optimize(i,j):
        
        
        
        
if __name__=="__main__":
    df=pd.read_csv("testSet.txt",sep="\s+",engine="python")
    dataSet=df.to_numpy()