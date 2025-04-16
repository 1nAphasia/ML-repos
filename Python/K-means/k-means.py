import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

iris=pd.read_csv("iris.csv",header=None)

iris=iris.to_numpy(copy=True)

temp=iris[:,:4]
X=np.array(temp,dtype="float32")
Y=iris[:,4]

Y[Y=="Iris-setosa"]=0
Y[Y=="Iris-versicolor"]=1
Y[Y=="Iris-virginica"]=2


print(Y)

class KMeans:
    def __init__(self,k=3,maxIter=100,tol=1e-4,randomState=None):
        self.k=k
        self.maxIter=maxIter
        self.tol=tol
        self.randomState=randomState
        self.centroids=None
        self.labels=None

    def initializeCentroids(self,X):
        np.random.seed(self.randomState)
        index=np.random.choice(len(X),self.k,replace=False) #随机抽取k个中心点
        return X[index]
    
    def computeDistances(self,X):
        p=X[:,np.newaxis]-self.centroids
        return np.linalg.norm(X[:,np.newaxis]-self.centroids,axis=2)
    
    def fit(self,X):
        self.centroids=self.initializeCentroids(X)

        for _ in range(self.maxIter):
            distances=self.computeDistances(X)
            self.labels=np.argmin(distances,axis=1)
            oldCentroids=self.centroids.copy()

            for i in range(self.k):
                clusterPoints=X[self.labels==i]
                if len(clusterPoints)>0:
                    self.centroids[i]=clusterPoints.mean(axis=0)

            if np.linalg.norm(self.centroids-oldCentroids)<self.tol:
                break

    def predict(self,X):
        distances=self.computeDistances(X)
        return np.argmin(distances,axis=1)


kmeans=KMeans(k=3,randomState=1)
kmeans.fit(X)
labels=kmeans.labels

fig=plt.figure(figsize=(12, 8))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(X[:, 0], X[:, 1],X[:, 2], c=Y, cmap='viridis', s=20)
ax.scatter(
    kmeans.centroids[:, 0], 
    kmeans.centroids[:, 1],
    kmeans.centroids[:, 2],
    c='red', marker='X', s=200, label='Centroids'
)
plt.title("K-means Clustering Result")
plt.legend()
plt.show()