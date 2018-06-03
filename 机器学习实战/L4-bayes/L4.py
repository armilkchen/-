# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:16:14 2018

@author: CHEN
"""
from numpy import*

import bayes
reload(bayes)
listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)

trainMat=[]
#trainMat [listOposts,myVocabList] 
#意义：myVocabList中的词在listOposts每行出现的情况
for postinDoc in listOPosts:
	trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))

p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)

reload(bayes)
bayes.testingNB()
