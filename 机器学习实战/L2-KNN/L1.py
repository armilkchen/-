#-*-coding:utf-8 -*-
import sys   #reload()之前必须要引入模块  
reload(sys)  
sys.setdefaultencoding('utf-8')

import kNN
group,labels = kNN.createDataSet()

print group,labels
output = kNN.classify0([0,0],group,labels,3)
print output



#读取文件函数
reload(kNN)
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')

#检测散点图函数
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
from numpy import *
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15*array(datingLabels), 15*array(datingLabels))#绘制散点图 [:,0]去所有行的第0个元素
plt.show()

#检测归一化函数
reload(kNN)
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)

#k = kNN.datingClassTest() #测试分类器


#kNN.classifyPerson() #预测


testVector = kNN.img2vector('testDigits/0_13.txt')
#print testVector[0,0:31]
#print testVector[0,32:63]


kNN.handwritingClassTest()