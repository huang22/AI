# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 13:24:30 2016

@author: huang
"""

from numpy import *
import operator
import csv
import scipy

# 把文件中的词汇与标签分好    
def mat_label(filename):
    fr = open(filename,'r')
    dataSize =  len(fr.readlines()) -1 
    fr.close()
    
    reader = csv.reader(open(filename))
    mat = []
    classLabel = zeros((dataSize,6))
    
    index = 0
    here = 0
    for DocumentID,Words,anger,disgust,fear,joy,sad,surprise in reader:
        if here == 0:
            here = 1
            continue
        
        lineMat = []
        Words = Words.strip()
        listFromLine = Words.split(' ')
        mat.append(listFromLine)
        
        lineMat.append(float(anger))
        lineMat.append(float(disgust))
        lineMat.append(float(fear))
        lineMat.append(float(joy))
        lineMat.append(float(sad))
        lineMat.append(float(surprise))
        
        classLabel[index,:] = lineMat
        index += 1
        
        
    return mat,classLabel,dataSize

# 做成词汇表
def vocabulary():
    mat1,classLabel1,len1 = mat_label('Dataset_train.csv')
    mat2,classLabel2,len2 = mat_label('Dataset_validation.csv')
    reader = csv.reader(open("Dataset_test.csv"))
    mat3 = []

    here = 0
    for DocumentID,Words,anger,disgust,fear,joy,sad,surprise in reader:
        if here == 0:
            here = 1
            continue
        Words = Words.strip()
        listFromLine = Words.split(' ')
        mat3.append(listFromLine)
    
    voca = []
    
    for item in mat1:  
        for i in item:
            if not i in voca:
                voca.append(i)
                
    for item in mat2:  
        for i in item:
            if not i in voca:
                voca.append(i)
        
    for item in mat3:  
        for i in item:
            if not i in voca:
                voca.append(i)
            
    return voca,len1
 
# 做one hot矩阵   
def one_hot(filename,voca,length):
    mat,classLabel,lineOfMat = mat_label(filename)
      
    oneMat = zeros((lineOfMat,len(voca)))
    
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
    mat,classLabel,lineOfMat = mat_label(filename)
      
    TFMat = zeros((lineOfMat,len(voca)))
    
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
    mat,classLabel,lineOfMat = mat_label(filename)
      
    TF_IDFMat = zeros((lineOfMat,len(voca)))
    
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
    
    here = 0
    
    # 计算k个最近邻共同作用得到的值
    Label = labels[sortedIndex[0]] / distances[0] 

    for i in range(1,k):
        Label += labels[sortedIndex[i]] / distances[i]
        
    # 所有情感值总和为1
    return Label / Label.sum(axis=0)

# 计算两个list相乘
def XmulY(X,Y):
    newList=[]
    for i in range(len(X)):
        newList.append(X[i] * Y[i])
    return newList

# 计算两个序列的相关系数
def correlation(X,Y):
    cov = mean(XmulY(X,Y))-mean(X) * mean(Y)
    varX = var(X)
    varY = var(Y)
    
    return cov / sqrt(double(varX * varY))
    
# 从kNN得到测试集所有样本标签，并且计算正确率    
def test(matType):
    
    voca,length = vocabulary()

    if matType == "TF_IDF":
           trainMat,trainLabel,trainOneMat = TF_IDF('Dataset_train.csv',voca,length)
           testMat,testLabel,testOneMat = TF_IDF('Dataset_validation.csv',voca,length)
    if matType == "TF":
           trainMat,trainLabel,trainOneMat = TF('Dataset_train.csv',voca,length)
           testMat,testLabel,testOneMat = TF('Dataset_validation.csv',voca,length)
    if matType == "one_hot":
           trainMat,trainLabel,trainOneMat = one_hot('Dataset_train.csv',voca,length)
           testMat,testLabel,testOneMat = one_hot('Dataset_validation.csv',voca,length)
        
    
    # 预测所有测试样本的label
    myLabel = zeros((len(testOneMat),6))
    #print kNN(testOneMat[105],trainOneMat,trainLabel,1)

    index = 0
    for i in testOneMat:
        myLabel[index,:]=(kNN(i,trainOneMat,trainLabel,5))
        index += 1
    
    myLabelT= transpose(myLabel)
    testLabelT=transpose(testLabel)
    index = 0
    corrSum = 0
    for i in myLabelT:
        corrSum += correlation(i,testLabelT[index])
        print correlation(i,testLabelT[index])
        index+=1
    return corrSum/double(index)