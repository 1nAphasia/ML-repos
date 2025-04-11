import numpy as np
import matplotlib.pyplot as plt
import random

def loadDataset():
    dataset=[]
    labels=[]
    fr=open("./data/testSet.txt")
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataset.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labels.append(float(lineArr[2]))
    fr.close()
    return dataset,labels

def sigmoid(x):
    return 1/(1+np.exp(-x))
        
def gradAscent(dataset,labels,learningRate=0.01,maxIter=500):
    dataMat=np.asmatrix(dataset)
    labelMat=np.asmatrix(labels).transpose()
    m,n=np.shape(dataMat)
    alpha=learningRate
    weights=np.ones((n,1))
    for j in range(maxIter):
        indices = list(range(m)) 
        np.random.shuffle(indices)
        for i in indices:
            alpha_j = alpha / (1 + j)  
            h = sigmoid(dataMat[i] * weights) 
            error = labelMat[i] - h  
            weights = weights + alpha_j * dataMat[i].T * error 
    return weights.getA()

def plotBestFit(weights):
    dataMat,labelMat=loadDataset()
    dataArr=np.array(dataMat)
    n=np.shape(dataMat)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=20,c='red',marker="s",alpha=.5)
    ax.scatter(xcord2,ycord2,s=20,c='green',alpha=.5)
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.title('BestFit')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

        
if __name__=="__main__":
    dataset,labels=loadDataset()
    plotBestFit(gradAscent(dataset,labels))