# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:02:31 2018

@author: CHEN
"""

import sys   #reload()之前必须要引入模块  
reload(sys)  
sys.setdefaultencoding('utf-8')

import trees

reload(trees)
myDat,labels=trees.createDataSet()
#myDat[0][-1]='maybe'
#print myDat
#print trees.calcShannonEnt(myDat)
reload(trees)
#print trees.splitDataSet(myDat,0,1)
#print trees.splitDataSet(myDat,0,0)

dataSet = [[1, 1, 'yes'], 
[1, 1, 'yes'],
[1, 0, 'no'],
[0, 1, 'no'],
[0, 1, 'no']]
#print dataSet

import treePlotter
reload(treePlotter)
myTree=treePlotter.retrieveTree(0)
#treePlotter.createPlot(myTree)

myTree['no surfacing'][3]='maybe'
#print myTree
#treePlotter.createPlot(myTree)

a = trees.classify(myTree,labels,[1,0])
print a

trees.storeTree(myTree,'classifierStorage.txt')
trees.grabTree('classifierStorage.txt')

