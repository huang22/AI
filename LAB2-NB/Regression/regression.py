# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:51:35 2016

@author: huang
"""

from numpy import *
import operator
import csv
import scipy

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
# 把文件中的词汇与标签分好    
def mat_label_reg(filename):
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
        
    return mat,classLabel
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
# 做成词汇表
def reg_vocabulary():
    mat1,classLabel1 = mat_label_reg('Dataset_train.csv')
    mat2,classLabel2 = mat_label_reg('Dataset_validation.csv')
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
            
    return voca,len(classLabel1),mat1,mat3

# 制作停用词
def stop_word():
    
    voca,length,mat1,mat3 = reg_vocabulary()

    trainMat,trainLabel,trainOneMat = TF_IDF('Dataset_train.csv',voca,length,"regression")
    
    rela = zeros((6,len(trainOneMat[0])))  
    # 计算相关度
    for i in range(len(trainOneMat)):
        for j in range(6):
            rela[j] += trainLabel[i][j] * trainOneMat[i]

    re = rela.T
    index = 0
    stop_word = [] 
    for i in range(len(re)):
        # 相关度在每个情感都是比较大的，则为停用词
        if((re[i][0] > 0.3 and re[i][1] > 0.3 and re[i][2] > 0.3 and re[i][3] > 0.3 and re[i][4] > 0.3 and re[i][5] > 0.3)):
            stop_word.append(voca[i])
            index += 1
    print stop_word
    print index
    new_voca = []
    # 新的词汇表
    for i in voca:
        if(i not in stop_word):
            new_voca.append(i)
    return new_voca
    
# 做one hot矩阵   
def one_hot(filename,voca,length,testType):
    if(testType == "classification"):
        mat,classLabel = mat_label(filename)
    else:
        mat,classLabel = mat_label_reg(filename)
      
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
def TF(filename,voca,length,testType):
    if(testType == "classification"):
        mat,classLabel = mat_label(filename)
    else:
        mat,classLabel = mat_label_reg(filename)
      
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
def TF_IDF(filename,voca,length,testType):
    if(testType == "classification"):
        mat,classLabel = mat_label(filename)
    else:
        mat,classLabel = mat_label_reg(filename)
      
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
    
# 用分类方法做回归，进行NB计算
def trainNB_cla_reg(dataSet, labels):
    
    dataSize = len(dataSet[0])
    fileSize = len(dataSet)
    # 每个训练样例分类
    pp = labels.argsort() + 1
    pp = pp.T
    maLabel = []
    for i in range(len(pp[0])):
        maLabel.append(pp[0][i])
    pEmotion = zeros((6,1))
    
    pNum = zeros((6,dataSize))
    pDenom = zeros((6,1))
    # 情感词袋
    for i in range(fileSize):
        pNum[(maLabel[i])-1] += dataSet[i]
        pDenom[(maLabel[i])-1] += sum(dataSet[i])
    
    # 多项式模型
    for i in range(6):
        pEmotion[i] = (float(pDenom[i]) / float(pDenom.sum()))
    # 伯努利模型
    '''
    for i in range(6):
        pEmotion[i] = (float(maLabel.count((i+1))) / float(fileSize))
    '''
    
    return maLabel,pDenom,pNum
    
# 回归，进行NB计算
def trainNB_reg(dataSet, labels):
    
    dataSize = len(dataSet[0])
    fileSize = len(dataSet)
    
    pEmotion = zeros((6,1))
    
    pNum = zeros((6,dataSize))
    pDenom = zeros((6,1))
    # 计算每一类情感的每个词总数以及总词数
    for i in range(fileSize):
        pNum += dataSet[i]
        pDenom += sum(dataSet[i])
        
    # 急计算每种情绪的概率
    for i in range(6):
        pEmotion[i] = (float(pNum[i].sum()) / float(pDenom.sum()))
    
    
    return pEmotion,pDenom,pNum

# 分类，进行NB计算
def trainNB_cla(dataSet, labels):
    
    dataSize = len(dataSet[0])
    fileSize = len(dataSet)
    
    pEmotion = zeros((6,1))
    
        
    pNum = zeros((6,dataSize))
    pDenom = zeros((6,1))
        
    for i in range(fileSize):
        pNum[int(labels[i])-1] += dataSet[i]
        pDenom[int(labels[i])-1] += sum(dataSet[i])
    # 伯努利模型 
    '''
    for i in range(6):
        pEmotion[i] = (float(labels.count(str(i+1))) / float(fileSize))
    '''
    # 多项式模型
    for i in range(6):
        pEmotion[i] = (float(pDenom[i]) / float(pDenom.sum()))
    
    return pEmotion,pDenom,pNum
# 从NB得到测试集所有样本标签，并且计算正确率    
def test_poly(matType = "one_hot"):
    
    dddd,length,mat1,mat3 = reg_vocabulary()
    voca = stop_word()
    oneMat = zeros((len(mat3),len(voca)))
    

    if matType == "TF_IDF":
           trainMat,trainLabel,trainOneMat = TF_IDF('Dataset_train.csv',voca,length,"regression")
           testMat,testLabel,testOneMat = TF_IDF('Dataset_validation.csv',voca,length,"regression")
    if matType == "TF":
           trainMat,trainLabel,trainOneMat = TF('Dataset_train.csv',voca,length,"regression")
           testMat,testLabel,testOneMat = TF('Dataset_validation.csv',voca,length,"regression")
    if matType == "one_hot":
           trainMat,trainLabel,trainOneMat = one_hot('Dataset_train.csv',voca,length,"regression")
           testMat,testLabel,testOneMat = one_hot('Dataset_validation.csv',voca,length,"regression")
    myLabel = zeros((len(testOneMat),6))
    #test的one_hot/TF/TF_IDF矩阵
    if matType == "one_hot":
        index = 0
        for item in mat3:
            lineMat = []
            for i in voca:
                if i in item:
                    lineMat.append(item.count(i))
                else:
                    lineMat.append(0)
            oneMat[index,:] = lineMat
            index += 1
    
    if matType == "TF":
        index = 0
        for item in mat3:
            lineMat = []
            for i in voca:
                if i in item:
                    lineMat.append(double(item.count(i)) / double(len(item)))
                else:
                    lineMat.append(0)
            oneMat[index,:] = lineMat
            index += 1
    if matType == "TF_IDF":
        wordInFile = []
        for i in voca:
            tmp = 1
            for item in mat3:
                if i in item:
                    tmp += 1
            wordInFile.append(double(length)/double(tmp))
                
        
        index = 0
        for item in mat3:
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
            
    maLabel,pDenom,pNum = trainNB_cla_reg(trainOneMat,trainLabel)
    

    #分类算法直接用到回归
    '''
    index = 0
    for i in oneMat:
        pFinalEmo = zeros((6,1))
        for k in range(6):
            pFinalEmo[k] = pEmotion[k]

        for j in range(len(i)):
                if(i[j] != 0):
                    for k in range(6):
                        # 词数 + 1 / 总次数 + V
                        pFinalEmo[k] = float(pFinalEmo[k]) * (float(pNum[k][j] + 1) / float(pDenom[k] + i.sum()))

        pFinalEmo = pFinalEmo / pFinalEmo.sum(axis=0)
        myLabel[index] = (pFinalEmo.T)[0]
        index += 1
    print myLabel
    '''
    
    # regression每一行训练样例得到结果相加
    index = 0
    for i in oneMat:
        pFinalEmo = zeros((6,1))
        labelIndex = 0
        # 每一行测试样例都要从所有训练样例中计算情感值
        for k in trainOneMat:
            pEmotion = zeros((6,1))
            # 计算该训练样例的各个情感值
            for ll in range(len(trainLabel[labelIndex])):
                pEmotion[ll][0] = trainLabel[labelIndex][ll]
            labelIndex += 1
            for j in range(len(i)):
                if(i[j] != 0):
                    for l in range(6):
                        # 词数 + 1 / 总词数 + V
                        pEmotion[l] = float(pEmotion[l]) * (float(k[j]+1)) /  (float(k.sum() + pDenom.sum()))
            pFinalEmo += pEmotion
        # 归一化
        pFinalEmo = pFinalEmo / pFinalEmo.sum(axis=0)
        myLabel[index] = (pFinalEmo.T)[0]
        index += 1
    print myLabel
    
    for i in range(len(testLabel)):
        for j in range(len(testLabel[0])):
            testLabel[i][j] = float(testLabel[i][j])
    
    testLabelT=transpose((testLabel))
    myLabelT= transpose(myLabel)
    index = 0
    corrSum = 0
    for i in myLabelT:
        corrSum += correlation(i,testLabelT[index])
        print correlation(i,testLabelT[index])
        index+=1
    print myLabel
    
    for i in range(len(testLabel)):
        for j in range(len(testLabel[0])):
            testLabel[i][j] = float(testLabel[i][j])
    
    #计算相关系数
    testLabelT=transpose((testLabel))
    myLabelT= transpose(myLabel)
    index = 0
    corrSum = 0
    for i in myLabelT:
        corrSum += correlation(i,testLabelT[index])
        print correlation(i,testLabelT[index])
        index+=1
    #写出labels文件
    writer = csv.writer(file('D_test.csv', 'wb'))
    aa =[]
    writer.writerow(aa)
    for a in myLabel:
        writer.writerow(a)
        
    return corrSum/double(index)
    
 
'''
#####################3
#########################
'''    
# 从NB得到测试集所有样本标签，并且计算正确率    
def test_Bernolli(matType = "one_hot"):
    
    voca,length,mat1,mat3 = reg_vocabulary()
    oneMat = zeros((len(mat3),len(voca)))

    if matType == "TF_IDF":
           trainMat,trainLabel,trainOneMat = TF_IDF('Dataset_train.csv',voca,length,"regression")
           testMat,testLabel,testOneMat = TF_IDF('Dataset_validation.csv',voca,length,"regression")
    if matType == "TF":
           trainMat,trainLabel,trainOneMat = TF('Dataset_train.csv',voca,length,"regression")
           testMat,testLabel,testOneMat = TF('Dataset_validation.csv',voca,length,"regression")
    if matType == "one_hot":
           trainMat,trainLabel,trainOneMat = one_hot('Dataset_train.csv',voca,length,"regression")
           testMat,testLabel,testOneMat = one_hot('Dataset_validation.csv',voca,length,"regression")
    myLabel = zeros((len(testOneMat),6))
    #test的one_hot/TF/TF_IDF矩阵
    if matType == "one_hot":
        index = 0
        for item in mat3:
            lineMat = []
            for i in voca:
                if i in item:
                    lineMat.append(item.count(i))
                else:
                    lineMat.append(0)
            oneMat[index,:] = lineMat
            index += 1
    
    if matType == "TF":
        index = 0
        for item in mat3:
            lineMat = []
            for i in voca:
                if i in item:
                    lineMat.append(double(item.count(i)) / double(len(item)))
                else:
                    lineMat.append(0)
            oneMat[index,:] = lineMat
            index += 1
    if matType == "TF_IDF":
        wordInFile = []
        for i in voca:
            tmp = 1
            for item in mat3:
                if i in item:
                    tmp += 1
            wordInFile.append(double(length)/double(tmp))
                
        
        index = 0
        for item in mat3:
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

    pEmotion,pDenom,pNum = trainNB_reg(trainOneMat,trainLabel)
    
    # 计算含有xk的文档数量
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
            # 文档数 + 1 / 总词数 + 2
            pFinalEmo[k] = float(tmp + 1) / float(pDenom[k] + 2)
                        
        pFinalEmo = pFinalEmo / pFinalEmo.sum(axis=0)
        myLabel[index] = (pFinalEmo.T)[0]
        index += 1
    
    for i in range(len(testLabel)):
        for j in range(len(testLabel[0])):
            testLabel[i][j] = float(testLabel[i][j])
    
    # 计算相关系数
    testLabelT=transpose((testLabel))
    myLabelT= transpose(myLabel)
    index = 0
    corrSum = 0
    for i in myLabelT:
        corrSum += correlation(i,testLabelT[index])
        print correlation(i,testLabelT[index])
        index+=1
    
    # 输出labels
    writer = csv.writer(file('Bernolli_vali.csv', 'wb'))
    aa =[]
    writer.writerow(aa)
    for a in myLabel:
        writer.writerow(a)
        
    return corrSum/double(index)
    
def test_class(matType = "one_hot"):
    
    voca,length = vocabulary()

    if matType == "TF_IDF":
           trainMat,trainLabel,trainOneMat = TF_IDF('train.txt',voca,length,"classification")
           testMat,testLabel,testOneMat = TF_IDF('test.txt',voca,length,"classification")
    if matType == "TF":
           trainMat,trainLabel,trainOneMat = TF('train.txt',voca,length,"classification")
           testMat,testLabel,testOneMat = TF('test.txt',voca,length,"classification")
    if matType == "one_hot":
           trainMat,trainLabel,trainOneMat = one_hot('train.txt',voca,length,"classification")
           testMat,testLabel,testOneMat = one_hot('test.txt',voca,length,"classification")
    
    myLabel = []

    # 计算每个情感概率，词数矩阵，总词数
    pEmotion,pDenom,pNum = trainNB_cla(trainOneMat,trainLabel)
    
    # 多项式模型
    for i in testOneMat:
        pFinalEmo = zeros((6,1))
        for k in range(6):
            pFinalEmo[k] = pEmotion[k]
        for j in range(len(i)):
                if(i[j] != 0):
                    for k in range(6):
                        # 出现该词次数 + 1 / 总词数 + V
                        pFinalEmo[k] = float(pFinalEmo[k]) * (float(pNum[k][j] + 1) / float(pDenom[k] + pDenom.sum()))
        
        pp = pFinalEmo.T.argsort()
        myLabel.append((pp[0][5] + 1))
    print myLabel
    '''
    # 伯努利模型
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
            # 出现该词文档数 + 1 / 总词数 + 2
            pFinalEmo[k] = float(tmp + 1) / float(pDenom[k] + 2)
                        
        pp = pFinalEmo.T.argsort()
        myLabel.append((pp[0][5] + 1))
        index += 1
    print myLabel
    '''
    # 求正确率
    sum = 0
    for i in range(len(myLabel)):
        if(myLabel[i] == (int)(testLabel[i])):
            sum += 1
    print float(sum) / float(len(myLabel))

