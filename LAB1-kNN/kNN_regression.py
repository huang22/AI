# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 16:44:21 2016

@author: huang
"""

from numpy import *
import operator
import csv
from scipy import spatial

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
    '''           
    for item in mat2:  
        for i in item:
            if not i in voca:
                voca.append(i)
    '''
    for item in mat3:  
        for i in item:
            if not i in voca:
                voca.append(i)
          
    return voca,len1,mat3
 
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
    voca,length,mat3 = vocabulary()
    wordInFile = []
    for i in voca:
        tmp = 1
        for item in mat:
            if i in item:
                tmp += 1
        for item in mat3:
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
        distances.append(-spatial.distance.cosine(testX,i))
        
    # 距离升序得到索引
    sortedIndex = argsort(distances)
    
    # 计算k个最近邻共同作用得到的值
    Label = labels[sortedIndex[0]] * abs(distances[0]) 

    for i in range(1,k):
        if distances[i] > 0:
            break
        Label += labels[sortedIndex[i]] * abs(distances[i])
        
    # 所有情感值总和为1
    return Label / Label.sum(axis=0)

# 进行kNN计算
def kNN(testX, dataSet, labels, k,disType):
    
    # 计算测试样本与训练样本集的距离
    if disType == 1:
        distances = abs((tile(testX, (dataSet.shape[0], 1)) - dataSet)).sum(axis=1)
    
    if disType == 2:
        dataMat = (tile(testX, (dataSet.shape[0], 1)) - dataSet) ** 2
        distances = dataMat.sum(axis=1) ** 0.5
    
    if disType == 3:
        distances = abs((tile(testX, (dataSet.shape[0], 1)) - dataSet)).max(axis=1)
    
    # 距离升序得到索引
    sortedIndex = argsort(distances)

    rangeDis = distances[sortedIndex[dataSet.shape[0] - 1]] - distances[sortedIndex[1]]
    
    distances = (distances - distances[sortedIndex[1]])/ rangeDis
    #distances = (distances  - mean(distances)) / var(distances)
    # 计算k个最近邻共同作用得到的值
   
    Label = labels[sortedIndex[0]] / (1.5 + distances[0] ** 5) 
    for i in range(1,k):
        Label += labels[sortedIndex[i]]  /(1.5 + distances[i] ** 5)
    
    #Label = labels[sortedIndex[k/2]]
    # 所有情感值总和为1
    return Label / Label.sum(axis=0)
    
def XplusY(X,Y):
    newList=[]
    for i in range(len(X)):
        newList.append(X[i] * Y[i])
    return newList
    
def correlation(X,Y):
    cov = mean(XplusY(X,Y))-mean(X) * mean(Y)
    varX = var(X)
    varY = var(Y)
    
    return cov / sqrt(double(varX * varY))
    
# 从kNN得到测试集所有样本标签，并且计算正确率    
def vali(kk,matType = "one_hot"):
    
    voca,length,mm = vocabulary()

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
   
    index = 0
    for i in testOneMat:
        myLabel[index,:]=(kNN(i,trainOneMat,trainLabel,kk,1))
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
    
def reg():
    for i in range(1,30):
        print "k = %d: %f" %(i, vali(i,"TF"))
        
def test():
    voca,length,mat = vocabulary()
    trainMat,trainLabel,trainOneMat = one_hot('Dataset_train.csv',voca,length)
    
    oneMat = zeros((len(mat),len(voca)))
    

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
       
    
    '''
    index = 0
    for item in mat:
        lineMat = []
        for i in voca:
            if i in item:
                lineMat.append(double(item.count(i)) / double(len(item)))
            else:
                lineMat.append(0)
        oneMat[index,:] = lineMat
        index += 1
    
    
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
        oneMat[index,:] = lineMat
        index += 1    
    '''

    myLabel = zeros((len(oneMat),6))
    
    index = 0
    for i in oneMat:
        myLabel[index,:]=(kNN(i,trainOneMat,trainLabel,20,1))
        index += 1
 
    writer = csv.writer(file('Dataset.csv', 'wb'))
    aa =[]
    writer.writerow(aa)
    for a in myLabel:
        writer.writerow(a)
        
    #return myLabel