import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import operator


def classify0(testVec,datingdataMat,datingLabels,k):
    m=datingdataMat.shape[0]
    testMat=np.tile(testVec,(m,1))
    diffMat=testMat-datingdataMat
    diffMat=diffMat**2
    sqDist=diffMat.sum(1)
    dist=sqDist**0.5
    sortedIndex=np.argsort(dist)
    classCount={}
    for i in range(k):
        votedLabel=datingLabels[sortedIndex[i]]
        classCount[votedLabel]=classCount.get(votedLabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def file2Matrix(filename):
    fr=open(filename)
    lineList=fr.readlines()
    lineNum=len(lineList)
    returnMat=np.zeros((lineNum,3))
    classLabelVector=[]
    index=0
    for line in lineList:
        line=line.strip()
        listFromline=line.split('\t')
        returnMat[index,:]=listFromline[0:3]
        if(listFromline[-1])=="didntLike":
            classLabelVector.append(1)
        elif listFromline[-1]=="smallDoses":
            classLabelVector.append(2)
        elif listFromline[-1]=="largeDoses":
            classLabelVector.append(3)
        index+=1
    return returnMat,classLabelVector
    
def showdatas(datingDataMat,datingLabels):

    plt.rcParams['font.family'] = 'SimHei'
    fig,axs=plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False,figsize=(13,8))

    numberOfLabels=len(datingLabels)
    labelsColors=[]
    for i in datingLabels:
        if i==1:
            labelsColors.append("black")
        if i==2:
            labelsColors.append("orange")
        if i==3:
            labelsColors.append("red")

    axs[0][0].scatter(x=datingDataMat[:,0],y=datingDataMat[:,1],color=labelsColors,s=15,alpha=.5)
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    axs[0][1].scatter(x=datingDataMat[:,0],y=datingDataMat[:,2],color=labelsColors,s=15,alpha=.5)
    axs0_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    axs0_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数')
    axs0_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    axs[1][0].scatter(x=datingDataMat[:,1],y=datingDataMat[:,2],color=labelsColors,s=15,alpha=.5)
    axs0_title_text = axs[1][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    axs0_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比')
    axs0_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show()

def autoNorm(dataset):
    minVals=dataset.min(0)
    maxVals=dataset.max(0)
    ranges=maxVals-minVals
    normDataset=np.zeros(np.shape(dataset))
    m=dataset.shape[0]
    normDataset=dataset-np.tile(minVals,(m,1))
    normDataset=normDataset/np.tile(ranges,(m,1))
    return normDataset,ranges,minVals

def datingClassTest():
    filename="./Dataset/datingTestSet.txt"
    datingDataMat,datingLabels=file2Matrix(filename)
    hoRatio=0.10
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
        print("分类结果:%d\t真实类别:%d"%(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print("错误率:%f%%"%(errorCount/float(numTestVecs)*100))

def classifyPerson():
    resultList=['讨厌','有点喜欢','非常喜欢']
    percentTats=float(input("玩视频游戏消耗时间百分比:"))
    ffMiles=float(input("每年获得的飞行常客里程数:"))
    iceCream=float(input("每周消费的冰淇淋公升数:"))
    filename="./Dataset/datingTestSet.txt"
    datingDataMat,datingLabels=file2Matrix(filename)
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inarr=np.array([ffMiles,percentTats,iceCream])
    norminArr=(inarr-minVals)/ranges
    classifierResult=classify0(norminArr,normMat,datingLabels,3)
    print("你可能%s这个人"%(resultList[classifierResult-1]))

if __name__ =="__main__":
    # filename="./Dataset/datingTestSet.txt"
    # datingDataMat,datingLabels=file2Matrix(filename)
    # showdatas(datingDataMat,datingLabels)
    # normDataset,ranges,minVals=autoNorm(datingDataMat)
    classifyPerson()