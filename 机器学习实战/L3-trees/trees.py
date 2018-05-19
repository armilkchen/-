# -*- coding: utf-8 -*-
"""
Created on Fri May 11 15:31:07 2018

@author: CHEN
"""

from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #len返回dataset的数目，长度
    labelCounts = {} #创建字典 计算标签出现的次数
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1] #dataset每一列的最后一个元素赋予currentLabel 即每一行数据的最后一个数据代表的是标签
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0 
        labelCounts[currentLabel] += 1
        #如果currentlabel不在labelcounts字典的key中 则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
        
    # 对于 label 标签的占比，求出 label 标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries #标签类别出现的概率
        shannonEnt -= prob * log(prob,2) # 以2为底求对数 计算香农熵
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
        splitDataSet(通过遍历dataSet数据集，求出axis对应的colnum列的值为value的行)
        就是依据axis列进行分类，如果axis列的数据等于 value的时候，就要将 axis 划分到我们创建的新的数据集中
    Args:
        dataSet 数据集                 待划分的数据集
        axis 表示每一行的axis列        划分数据集的特征
        value 表示axis列对应的value值   需要返回的特征的值。
    Returns:
        axis列为value的数据集【该数据集需要排除axis列】！！！
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:     # 判断axis列的值是否为value
            reducedFeatVec = featVec[:axis]     # [:axis]表示前axis列
            reducedFeatVec.extend(featVec[axis+1:])# [axis+1:]表示从跳过 axis 的 axis+1列，取接下来的数据
            retDataSet.append(reducedFeatVec)
            '''

            music_media.append(object) 向列表中添加一个对象object
            music_media.extend(sequence) 把一个序列seq的内容添加到列表中 (跟 += 在list运用类似， music_media += sequence)
            1、使用append的时候，是将object看作一个对象，整体打包添加到music_media对象中。
            2、使用extend的时候，是将sequence看作一个序列，将这个序列和music_media序列合并，并放在其后面。
            music_media = []
            music_media.extend([1,2,3])
            print music_media
            #结果：
            #[1, 2, 3]
            
            music_media.append([4,5,6])
            print music_media
            #结果：
            #[1, 2, 3, [4, 5, 6]]
            
            music_media.extend([7,8,9])
            print music_media
            #结果：
            #[1, 2, 3, [4, 5, 6], 7, 8, 9]
            '''
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels 最后一列是标签 去掉
    baseEntropy = calcShannonEnt(dataSet)  #计算数据集的原始信息熵
    bestInfoGain = 0.0; bestFeature = -1  # 最优的信息增益值, 和最优的Featurn编号
    for i in range(numFeatures):        #iterate over all the features 迭代所有numfeatures
        featList = [example[i] for example in dataSet]#遍历所有的元素
        #将dataSet中的数据按行依次放入example中，然后取得example中的example[i]元素，放入列表featList中
        uniqueVals = set(featList)       #获得每个唯一属性值划分的数据集
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)#对每个唯一属性值划分一次数据集
            prob = len(subDataSet)/float(len(dataSet)) #得到每个唯一属性值占总属性的概率
            newEntropy += prob * calcShannonEnt(subDataSet)      #每一个属性占比*香农熵 再累计相加
        # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
        # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy 原始信息熵减去新的
        #不断比较 得到差值最大的
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer



def majorityCnt(classList):#投票表决 返回出现次数最多的分类
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #operator.itemgetter（1） 根据第一个域进行排序 函数获取的不是值，而是定义了一个函数，通过该函数作用到对象上才能获取值。
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #将dataSet中的数据按行依次放入example中，然后取得example中的example[-1]元素，放入列表classList中
    if classList.count(classList[0]) == len(classList): # count() 函数是统计括号中的值在list中出现的次数
        #如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
        return classList[0]# 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
     # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
        return majorityCnt(classList)
    
    #开始绘制树
    bestFeat = chooseBestFeatureToSplit(dataSet)# 选择最优的列，得到最优列对应的label含义
    bestFeatLabel = labels[bestFeat]# 获取label的名称
    myTree = {bestFeatLabel:{}}#初始化
    del(labels[bestFeat])#del删除
    featValues = [example[bestFeat] for example in dataSet]# 取出最优列，然后它的branch做分类
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        # 求出剩余的标签label
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
    return myTree     

def classify(inputTree,featLabels,testVec):#使用决策树的分类函数
    """classify(给输入的节点，进行分类)

    Args:
        inputTree  决策树模型
        featLabels Feature标签对应的名称
        testVec    测试输入的数据
    Returns:
        classLabel 分类的结果值，需要映射label才能知道名称
    """
    firstStr = inputTree.keys()[0]# 获取tree的根节点对应的key值
    secondDict = inputTree[firstStr]# 通过key得到根节点对应的value
    featIndex = featLabels.index(firstStr)
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类 得知是在树的第几层
    key = testVec[featIndex] #获取测试数据那一层的数值
    valueOfFeat = secondDict[key]
    #检测是否为dict 其实是看分类是否结束
    if isinstance(valueOfFeat, dict): #判断实例是否是这个类或者object是变量 这里判断是不是字典
        classLabel = classify(valueOfFeat, featLabels, testVec)#如果是dict就重新执行函数
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
                       


