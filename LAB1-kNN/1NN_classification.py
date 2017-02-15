# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:35:11 2016

@author: huang
"""

from numpy import *
import operator

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
 
# 做one hot矩阵   
def one_hot(filename,voca,length):
    mat,classLabel = mat_label(filename)
      
    oneMat = zeros((len(classLabel),len(voca)))
    
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

# TF
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
    
# TF
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
    
# 进行kNN计算
def kNN(testX, dataSet, labels, k):
    
    # 计算测试样本与训练样本集的距离
    dataMat = (tile(testX, (dataSet.shape[0], 1)) - dataSet) ** 2
    distances = dataMat.sum(axis=1)
    
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

# 从kNN得到测试集所有样本标签，并且计算正确率    
def test(matType):
    
    voca,length = vocabulary()

    if matType == "TF_IDF":
           trainMat,trainLabel,trainOneMat = TF_IDF('train.txt',voca,length)
           testMat,testLabel,testOneMat = TF_IDF('test.txt',voca,length)
    if matType == "TF":
           trainMat,trainLabel,trainOneMat = TF('train.txt',voca,length)
           testMat,testLabel,testOneMat = TF('test.txt',voca,length)
    if matType == "one_hot":
           trainMat,trainLabel,trainOneMat = one_hot('train.txt',voca,length)
           testMat,testLabel,testOneMat = one_hot('test.txt',voca,length)
    
    myLabel = []
    
    # 预测所有测试样本的label
    for i in testOneMat:
        myLabel.append(kNN(i,trainOneMat,trainLabel,1))
    
    # 计算正确率
    corrSum = 0
    index = 0
    for i in myLabel:
        if i == testLabel[index]:
            corrSum += 1
        index += 1
    
    print double(corrSum)/double(index)