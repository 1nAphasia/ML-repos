import numpy as np
import operator

def createDataset():
    group =np.array([[1,101],[5,89],[108,5],[115,8]])
    labels=['爱情片','爱情片','动作片','动作片']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    dist=sqDiffMat.sum(axis=1)**0.5
    sortedDist=dist.argsort()
    classCount={}
    for i in range(k):
        votedLabel=labels[sortedDist[i]]
        classCount[votedLabel]=classCount.get(votedLabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
if __name__=="__main__":
    group,labels=createDataset()
    test=[101,20]
    test_class=classify0(test,group,labels,3)
    print(test_class)

