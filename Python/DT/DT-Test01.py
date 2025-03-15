from math import log
import numpy as np



def createDataSet():
    """     dataSet = [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    features=['年龄','有工作','有自己的房子','信贷情况','是否放贷']
    labels = ['不放贷', '放贷']             #分类属性 """
    dataSet = [
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
    ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
    ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
    ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
    ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
    ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
    ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']
    ]

    features = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "好瓜"]
    labels=[]
    return dataSet, features,labels                #返回数据集和分类属性
 

def calcEntropy(dataset):
    personNum=len(dataset)
    labelCount={}
    for vec in dataset:
        if vec[-1] not in labelCount:
            labelCount[vec[-1]]=0
        labelCount[vec[-1]]+=1  
    entropy=0.0
    for key in labelCount:
        p=labelCount[key]/personNum
        entropy-=p*log(p,2)
    return entropy

def chooseBestFeatureToSplit(dataset):
    featureNum=len(dataset[0])-1
    total=len(dataset)
    gainList=[]
    for i in range(featureNum):
        featureList=[]
        vecList=[]
        entropy=0.0
        for vec in dataset:
            if vec[i] not in featureList:
                featureList.append(vec[i])
        for feature in featureList:
            correctList=[]
            for vec in dataset:
                if vec[i]==feature:
                    correctList.append(vec)
            vecList.append(correctList)
        for vec in vecList:
            weight=len(vec)/total
            entropy+=weight*calcEntropy(vec)
            gain=calcEntropy(dataset)-entropy
        gainList.append(gain)
    max_index=np.argmax(gainList)
    print(gainList)
    return max_index

#统计axis上每一个元素的数量
def featureNumCheck(dataset,axis):
    featureCount={}
    for vec in dataset:
        featureCount[vec[axis]]=featureCount.get(vec[axis],0)+1
    return featureCount

def splitDataset(dataset,inFeatures,axis):
    featureList=[]
    for vec in dataset:
        if vec[axis] not in featureList:
            featureList.append(vec[axis])
    newDataset=[]
    for feature in featureList:
            featuredVec=[]
            for vec in dataset:
                if feature==vec[axis]:
                    reducedVec=vec[:axis]
                    reducedVec.extend(vec[axis+1:])
                    featuredVec.append(reducedVec)
            newDataset.append(featuredVec)
    newFeatures=inFeatures
    newFeatures.pop(axis)
    print(newFeatures,inFeatures)
    return newDataset,newFeatures,featureList

def treeGenerate(dataset,features):
    classCount=featureNumCheck(dataset,-1)
    print(classCount)
    if len(classCount)==1:
        return next(iter(classCount))
    if len(features)==0:
        return max(classCount,key=classCount.get)
    max_index=chooseBestFeatureToSplit(dataset)
    tree={'feature':features[max_index]}
    newDataset,newTargetFeatures,featureList=splitDataset(dataset,features,max_index)
    print("splited:",newDataset,newTargetFeatures)
    for i in range(len(featureList)):
        print()
        tree[featureList[i]]=treeGenerate(newDataset[i],newTargetFeatures.copy())
    return tree


if __name__=="__main__":
    
    dataset,features,labels=createDataSet()
    print(treeGenerate(dataset,features.copy()))



