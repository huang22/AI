# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 16:44:21 2016

@author: huang
"""

import numpy
import operator
from scipy import spatial

# 把文件中的词汇与标签分好    
def mat_label(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    num_lines = len(arrayLines)
    mat = []
    classLabel = []

    for line in arrayLines[1:]:
        line = line.strip()
        listFromLine = line.split(' ')
        mat.append(listFromLine[3:])
        classLabel.append(listFromLine[1])
        
    return mat,classLabel

# 做成词汇表
def vocabulary():
    mat1,classLabel1 = mat_label('train.txt')
    mat2,classLabel2 = mat_label('test.txt')
    voca = []
    
    for item in mat1:  
        for i in item:
            if not i in voca:
                voca.append(i)
                
    for item in mat2:  
        for i in item:
            if not i in voca:
                voca.append(i)

    return voca,len(classLabel1)
'''
###################三种类型的矩阵#####################
''' 
# 做one hot矩阵   
def one_hot(filename,voca,length):
    mat,classLabel = mat_label(filename)
      
    oneMat = zeros((len(classLabel),len(voca)))
    # 有则为1，无则为0
    index = 0
    for item in mat:
        lineMat = []
        for i in voca:
            if i in item:
                lineMat.append(1)
            else:
                lineMat.append(0)
        oneMat[index,:] = lineMat
        index += 1
        
    return mat,classLabel,oneMat

# TF矩阵
def TF(filename,voca,length):
    mat,classLabel = mat_label(filename)
      
    TFMat = zeros((len(classLabel),len(voca)))
    
    index = 0
    for item in mat:
        lineMat = []
        for i in voca:
            if i in item:
                lineMat.append(double(item.count(i)) / double(len(item)))
            else:
                lineMat.append(0)
        TFMat[index,:] = lineMat
        index += 1
        
    return mat,classLabel,TFMat
    
# TF_IDF矩阵
def TF_IDF(filename,voca,length):
    mat,classLabel = mat_label(filename)
      
    TF_IDFMat = zeros((len(classLabel),len(voca)))
    
    wordInFile = []
    for i in voca:
        tmp = 1
        for item in mat:
            if i in item:
                tmp += 1
        wordInFile.append(double(length)/double(tmp))
            
    
    index = 0
    for item in mat:
        lineMat = []
        tmp = 0
        for i in voca:
            if i in item:
                lineMat.append(log(wordInFile[tmp]) * double(item.count(i)) / double(len(item)))
            else:
                lineMat.append(0)
            tmp += 1
        TF_IDFMat[index,:] = lineMat
        index += 1
        
    return mat,classLabel,TF_IDFMat
    
'''
###################距离计算#####################
''' 
def cosDist(testX, dataSet, labels, k,disType):
    distances = []
    index = 0
    for i in dataSet:
        # 值比较大意味着距离近，为了升序，这里加上负号
        distances.append(-spatial.distance.cosine(testX,i))
        
    # 距离升序得到索引
    sortedIndex = argsort(distances)
    labelCount = {}
    
    # 计算k个最近邻
    for i in range(k):
        Label = labels[sortedIndex[i]]
        labelCount[Label] = labelCount.get(Label , 0) + 1
    
    # 输出k个最近邻的众数
    sortedLabelCount = sorted(labelCount.iteritems(), 
                              key=operator.itemgetter(1), reverse=True)    
                              
    return sortedLabelCount[0][0]

# 进行kNN计算
def kNN(testX, dataSet, labels, k,disType):
    
    # 曼哈顿距离
    if disType == 1:
        distances = abs((tile(testX, (dataSet.shape[0], 1)) - dataSet)).sum(axis=1)
    # 欧式距离
    if disType == 2:
        dataMat = (tile(testX, (dataSet.shape[0], 1)) - dataSet) ** 2
        distances = dataMat.sum(axis=1) ** 0.5
    #切比雪夫距离
    if disType == 3:
        distances = abs((tile(testX, (dataSet.shape[0], 1)) - dataSet)).max(axis=1)

       
   # 距离升序得到索引
    sortedIndex = argsort(distances)
    labelCount = {}
    
    rangeDis = distances[sortedIndex[dataSet.shape[0] - 1]] - distances[sortedIndex[0]]
    
    distances = (distances - distances[sortedIndex[0]])/ rangeDis
    
    # 计算k个最近邻
    
    for i in range(k):
        Label = labels[sortedIndex[i]]
        labelCount[Label] = exp(-distances[i]) + labelCount.get(Label , 0)
    '''
    for i in range(k):
        Label = labels[sortedIndex[i]]
        labelCount[Label] = 1.0/double(distances[i]) + labelCount.get(Label , 0)
     '''  
    # 输出k个最近邻的众数
    sortedLabelCount = sorted(labelCount.iteritems(), 
                              key=operator.itemgetter(1), reverse=True)    
                              
    return sortedLabelCount[0][0]
    
# 从kNN得到测试集所有样本标签，并且计算正确率    
def test():
    
    voca,length = vocabulary()
    trainMat,trainLabel,trainOneMat = TF('train.txt',voca,length)
    testMat,testLabel,testOneMat = TF('test.txt',voca,length)
    ##print cosDist(testOneMat[2],trainOneMat,trainLabel,1,4)    

    for kk in range(1,30):
        myLabel = []
        
        # 预测所有测试样本的label
        for i in testOneMat:
            myLabel.append(kNN(i,trainOneMat,trainLabel,kk,1))
        
        # 计算正确率
        corrSum = 0
        index = 0
        for i in myLabel:
            if i == testLabel[index]:
                corrSum += 1
            index += 1
        print "k = %d: %f" %(kk,double(corrSum)/double(index))
        #print double(corrSum)/double(index)
