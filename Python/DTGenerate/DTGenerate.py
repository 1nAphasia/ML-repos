import math

def CountTargetLabel(data,target_label):
    label_counts={}
    print(data)
    for feat_vec in data:
        current_label=feat_vec[target_label]
        print(current_label)
        label_counts[current_label]=label_counts.get(current_label,0)+1
    return label_counts

def CalcEntropy(data):
    label_counts=CountTargetLabel(data,-1)
    entropy=0.0
    for key in label_counts:
        prob=float(label_counts[key])/len(data)
        entropy-=prob*math.log(prob,2)
        return entropy;

def ChooseBestFeatureToSplit(data,labels):
    #选择最优划分特征
    label_gain={}
    dEntropy=CalcEntropy(data)
    for i in range(len(labels)):
        label_gain[i]=dEntropy-CalcSumEntDv(data,i)
    max_key=max(label_gain,key=label_gain.get)
    return max_key

def split_dataset(data,target_feat,labels):
    ret_dataset=[]
    ret_labels=[]
    for feat_vec in data:
        ret_dataset.append(feat_vec[:target_feat]+feat_vec[target_feat+1:])
    ret_labels.append(labels[:target_feat]+labels[target_feat+1:])
    return ret_dataset,ret_labels

def CalcSumEntDv(data,target_feat):
    label_counts=CountTargetLabel(data,target_feat)
    sum=0.0
    datalen=len(data)
    for key in label_counts:
        keydata=[]
        for vec in data:
            if(vec[target_feat]==key):
                keydata.append(vec)
        keylen=len(keydata)
        sum+=keylen/datalen*CalcEntropy(keydata)
    return sum

def majority_cnt(class_list):
    """多数表决，返回出现最多的类别"""
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]

def create_tree(data,labels):
    # 复刻数据集中已有的每一个类
    class_list=[example[-1] for example in data]
    # 如果所有的元素都是同一类别 直接返回第一个类就好了
    if class_list.count(class_list[0])==len(class_list):
        return class_list[0]
    # 如果用完所有特征仍无法划分,则返回出现最多的类别
    if len(data[0])==1:
        return majority_cnt(class_list)
    
    best_feat=ChooseBestFeatureToSplit(data,labels)
    best_feat_label=labels[best_feat]
    tree={best_feat_label:{}}
    del(labels[best_feat])
    feat_values=[example[best_feat] for example in data]
    unique_vals=set(feat_values)
    for value in unique_vals:
        sub_labels=labels[:]
        newdata,newlabels=split_dataset(data,best_feat,value)
        tree[best_feat_label][value]=create_tree(newdata,newlabels)
    return tree

data = [
    ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"],
    ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"],
    ["乌黑", "蜷缩", "浑响", "清晰", "凹陷", "硬滑", "是"],
    ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"],
    ["浅白", "蜷缩", "浑响", "清晰", "凹陷", "硬滑", "是"],
    ["青绿", "蜷缩", "浑响", "清晰", "稍凹", "软粘", "是"],
    ["乌黑", "蜷缩", "浑响", "清晰", "稍凹", "软粘", "是"],
    ["乌黑", "稍蜷", "浑响", "清晰", "稍凹", "硬滑", "是"],
    ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", "否"],
    ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", "否"],
    ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", "否"],
    ["青绿", "蜷缩", "浑响", "模糊", "平坦", "软粘", "否"],
    ["青绿", "稍蜷", "浑响", "稍糊", "凹陷", "硬滑", "否"],
    ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", "否"],
    ["乌黑", "稍蜷", "清晰", "稍糊", "稍凹", "软粘", "否"],
    ["浅白", "蜷缩", "浑响", "模糊", "稍凹", "硬滑", "否"],
    ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", "否"]
]

headers = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "好瓜"]

if __name__ == '__main__':
    my_tree = create_tree(data, headers)
    print("生成的决策树：", my_tree)