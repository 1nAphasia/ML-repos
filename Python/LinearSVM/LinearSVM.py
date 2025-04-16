import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


class SVM:
    def __init__(self, C=1.0, kernel='linear', tol=1e-3, max_iter=1000):
        self.C = C                      # 正则化参数
        self.ker = kernel               # 核函数（如'rbf'）
        self.tol = tol                  # 收敛容忍度
        self.max_iter = max_iter        # 最大迭代次数
        self.alpha = None               # 拉格朗日乘子
        self.b = 0                      # 偏置项
        self.X = None                   # 训练数据
        self.Y = None                   # 标签（±1）
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
        yi=self.Y[i]
        Ei=self.errors[i]
        ri=Ei*yi
        if(ri<-self.tol and self.alpha[i]<self.C) or (ri>self.tol and self.alpha[i]>0):
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
            
    
    def optimize(self,i,j):
        '''
        优化步骤：
        1.计算误差Ei=f(xi)-yi 这一步有误差缓存可以使用,所以直接取就可以,到后面再更新。
        2.计算上下界L和H
        3.计算常数eta
        4.更新alphaj
        5.根据上下界修建alphaj
        6.根据修剪过的alphaj更新alphai
        7.更新b1和b2
        8.根据b1和b2更新b
        '''

        if i==j :return False
        alphai_old=self.alpha[i]
        alphaj_old=self.alpha[j]
        yi=self.Y[i]
        yj=self.Y[j]

        if(yi!=yj):
            L=max(0,alphaj_old-alphai_old)
            H=min(self.C,self.C+alphaj_old-alphai_old)
        elif(yi==yj):
            L=max(0,alphaj_old+alphai_old-self.C)
            H=min(self.C,alphaj_old+alphai_old)
        
        tii=self.kernel(self.X[i],self.X[i])
        tij=self.kernel(self.X[i],self.X[j])
        tjj=self.kernel(self.X[j],self.X[j])

        eta=tii+tjj-2*tij
        if(eta<=1e-12):
            return False #eta会出现负值或0吗？

        alphaj_new=np.clip(alphaj_old+self.Y[j]*(self.errors[i]-self.errors[j])/eta,L,H)
        alphai_new=alphai_old+self.Y[i]*self.Y[j]*(alphaj_old-alphaj_new)

        self.alpha[i]=alphai_new
        self.alpha[j]=alphaj_new

        b1_new=self.b-self.errors[i]-self.Y[i]*(alphai_new-alphai_old)*tii-self.Y[j]*(alphaj_new-alphaj_old)*tij
        b2_new=self.b-self.errors[j]-self.Y[i]*(alphai_new-alphai_old)*tij-self.Y[j]*(alphaj_new-alphaj_old)*tjj

        if 0<alphai_new<self.C:
            self.b=b1_new
        elif 0<alphaj_new<self.C:
            self.b=b2_new
        else:
            self.b=(b1_new+b2_new)/2
        
        self.errors = [self.predict(self.X[k]) - self.Y[k] for k in range(len(self.X))]
        
        return True

    def kernel(self, x1, x2):
        if self.ker == 'linear':
            return np.dot(x1, x2)
        elif self.ker == 'rbf':
            gamma = 0.1  # 可设为参数
            return np.exp(-gamma * np.linalg.norm(x1-x2)**2)
        else:
            raise ValueError("Unsupported kernel")
    
    def predict(self,x):
        kernelVal=np.array([self.kernel(x_i,x) for x_i in self.X])
        return np.dot(self.alpha*self.Y,kernelVal)+self.b
        
        
        
        
if __name__=="__main__":
    df=pd.read_csv("testSet.txt",sep="\s+",engine="python")
    dataSet=df.to_numpy()
    svm=SVM()

    svm.fit(dataSet[:,0:-1],dataSet[:,-1])

    omega=np.sum(svm.alpha*svm.Y*svm.X.T,axis=1)
    slope=-omega[0]/omega[1]

    print(svm.alpha)
    print(omega,svm.b)

    plt.figure(figsize=(10,8))
    plt.scatter(dataSet[:,0],dataSet[:,1],c=dataSet[:,2],facecolors=None)


    plt.axline(xy1=(0,-svm.b/omega[1]),slope=slope,
               color="blue",
               linestyle="--")

    plt.show()