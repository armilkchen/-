# -*- coding: utf-8 -*-
"""
Created on Sat May 19 12:04:33 2018

@author: CHEN
"""

import sys   #reload()之前必须要引入模块  
reload(sys)  
sys.setdefaultencoding('utf-8')

import trees
import treePlotter

fr=open('lenses.txt') #打开数据
lenses=[inst.strip().split('\t') for inst in fr.readlines()]#strip去除首尾相应字符  split切片
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)

treePlotter.createPlot(lensesTree)