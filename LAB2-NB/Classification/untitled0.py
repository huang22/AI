# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:51:35 2016

@author: huang
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 20:46:13 2016

@author: huang
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from numpy import *
import operator

# 把文件中的词汇与标签分好    
def mat_label(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    mat = []
    classLabel = []
 
    for line in arrayLines[1:]:
        line = line.strip()
        listFromLine = line.split(' ')
        mat.append(listFromLine[3:])
        classLabel.append(listFromLine[1])
        
    return mat,classLabel

# 把文件中的词汇与标签分好    
def mat_label1(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    mat = []
 
    for line in arrayLines[1:]:
        line = line.strip()
        listFromLine = line.split(' ')
        mat.append(listFromLine[3:])
        
    return mat
    
# 做成词汇表
def vocabulary():
    mat1,classLabel1 = mat_label('train.txt')
    mat2 = mat_label1('test.txt')
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
                lineMat.append(item.count(i))
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
    
# 进行NB计算
def trainNB(dataSet, labels):
    
    dataSize = len(dataSet[0])
    fileSize = len(dataSet)
    
    pEmotion = zeros((6,1))
    
        
    pNum = zeros((6,dataSize))
    pDenom = zeros((6,1))
        
    for i in range(fileSize):
        pNum[int(labels[i])-1] += dataSet[i]
        pDenom[int(labels[i])-1] += sum(dataSet[i])
    '''
    for i in range(6):
        pEmotion[i] = (float(labels.count(str(i+1))) / float(fileSize))
    '''
    for i in range(6):
        pEmotion[i] = (float(pDenom[i]) / float(pDenom.sum()))
    
    return pEmotion,pDenom,pNum

# 从kNN得到测试集所有样本标签，并且计算正确率    
def test(matType = "one_hot"):
    
    voca,length = vocabulary()
    #print voca
    print len(voca)
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

    temp = trainOneMat.sum(axis = 0)
    wordCount = 0
    for i in temp:
        if(i != 0):
            wordCount += 1
    print wordCount
    # 预测所有测试样本的label
    pEmotion,pDenom,pNum = trainNB(trainOneMat,trainLabel)
    
    for i in testOneMat:
        pFinalEmo = zeros((6,1))
        for k in range(6):
            pFinalEmo[k] = pEmotion[k]

        for j in range(len(i)):
                if(i[j] != 0):
                    for k in range(6):
                        pFinalEmo[k] = float(pFinalEmo[k]) * (float(pNum[k][j] + 1) / float(pDenom[k] + wordCount))
        
        pp = pFinalEmo.T.argsort()
        myLabel.append((pp[0][5] + 1))
    print myLabel
    '''
    index = 0                        
    wordInFile = []
    for item in testOneMat:
        tmp = 0
        pFinalEmo = zeros((6,1))
        for k in range(6):
            pFinalEmo[k] = pEmotion[k]
        for i in range(len(item)):
            for j in trainOneMat:
                if (item[i] == j[i] and item[i] == 1):
                    tmp += 1
        for k in range(6):
            pFinalEmo[k] = float(tmp + 1) / float(pDenom[k] + 2)
                        
        pp = pFinalEmo.T.argsort()
        myLabel.append((pp[0][5] + 1))
        index += 1
    print myLabel
    '''
    sum = 0
    for i in range(len(myLabel)):
        if(myLabel[i] == (int)(testLabel[i])):
            sum += 1
    print float(sum) / float(len(myLabel))
