#coding:utf-8

from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,1.0]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):  #inX用于分类的输入向量 dataSet输入的训练样本集 labels标签向量 k表示选择最近邻居的数目
    dataSetSize = dataSet.shape[0]  #shape函数得矩阵的纬度（X,Y） shape[0]得X，shepe[1]得Y
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #tile的功能是重复某个数组。比如tile(A,n)，功能是将数组A重复n次，构成一个新的数组 默认是列重复 tile(A,(1,2))行1次，列2次
    sqDiffMat = diffMat**2  #**为乘方 **2为2次方
    sqDistances = sqDiffMat.sum(axis=1) #sum为求和 axis=0，则沿着纵轴进行操作；axis=1，则沿着横轴进行操作。
    distances = sqDistances**0.5 #开方
    #以上是计算距离
    sortedDistIndicies = distances.argsort() #  得到输入向量与训练样本每个的距离进行从小到大排序  argsort返回‘索引值！’
    classCount={}  #记录类别次数的字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] # 得到k所代表的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #计算该标签的数量 get（A，0）key=A 默认值为0
        sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
        #sorted进行排序
        #iteritems得到的是一个迭代器，item的得到的是list
        #itemgetter（1）根据第二个域的值进行排序
        # reverse = True 降序 ， reverse = False 升序（默认）
        return sortedClassCount[0][0]
    
def file2matrix(filename): #将文本记录转换为Numpy的解析程序
    fr = open(filename) #打开文本文件
    arrayOLines = fr.readlines() #读取的内容形成一个数组
    numberOfLines = len(arrayOLines)#返回数组长度 获取有多少行
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in arrayOLines:
        line = line.strip()#去除所有回车字符
        listFromLine = line.split('\t')#根据'\t'分隔符进行切片
        returnMat[index,:] = listFromLine[0:3]#将数据前三列提取出来,存放到returnMat的NumPy矩阵中   ###也就是“特征矩阵”
        classLabelVector.append(int(listFromLine[-1]))#将listFromLine最后一个增加到classLabelVector中
        index += 1
    return returnMat,classLabelVector


def autoNorm(dataSet): #归一化特征值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1)) # 生成与最小值之差组成的矩阵
    normDataSet = normDataSet/tile(ranges, (m,1))# 将最小值之差除以范围组成矩阵       #element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():  #分类器针对约会网站的测试代码
    hoRatio = 0.50      # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       # 从文件中加载数据
    normMat, ranges, minVals = autoNorm(datingDataMat) #归一化处理
    m = normMat.shape[0] #获取行数
    numTestVecs = int(m*hoRatio) # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
    
    
    
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    
    percentTats = float(raw_input("percentage of time spent playing video games ?"))#输入
    ffMiles = float(raw_input("frequent filer miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)#归一化处理
    inArr = array([ffMiles, percentTats, iceCream])#将获得数据集合成array
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels, 3)
    print "You will probably like this person: ", resultList[classifierResult - 1]
    
    
    
def img2vector(filename):#将图像转换为测试向量 格式转换？
    returnVect = zeros((1,1024))#创建1x1024零向量
    fr = open(filename)#打开文件
    for i in range(32):#按行读取
        lineStr = fr.readline()
        for j in range(32):#每一行的前32个元素依次添加到returnVect中
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():#使用k-近邻算法识别手写数字
    hwLabels = [] #
    trainingFileList = listdir('trainingDigits')        #返回trainingDigits目录下的文件名 返回的是文件名   #load the training set
    m = len(trainingFileList) #返回文件夹下文件的个数
    trainingMat = zeros((m,1024)) #m行1024的矩阵 每行矩阵都是一张图
    for i in range(m):
        fileNameStr = trainingFileList[i] 
        fileStr = fileNameStr.split('.')[0]     #take off .txt #去除文件名后面的.txt
        classNumStr = int(fileStr.split('_')[0]) #文件名都是类似1_45 获得数字的分类
        hwLabels.append(classNumStr) #将类别添加到hwLabels
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr) #将样本文件夹的文件转换格式 保存到trainingMat中
    testFileList = listdir('testDigits')        #iterate through the test set 遍历测试集  #返回testDigits目录下的文件名 返回的是文件名
    errorCount = 0.0
    mTest = len(testFileList) #返回文件夹下文件的个数
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt #去除文件名后面的.txt
        classNumStr = int(fileStr.split('_')[0])#文件名都是类似1_45 获得数字的分类
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr) #将测试文件夹的文件转换格式
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  #使用k-近邻算法分类 因为数字都0或1 不需要归一化
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    
    