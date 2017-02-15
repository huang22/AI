# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:16:37 2016

@author: huang
"""

from numpy import *
import numpy
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.pyplot as plt
import time
import re

# 训练数据
def trainData():
    fr_data = open('train_data.txt')
    arrayLines = fr_data.readlines()
    myMat = zeros((len(arrayLines),10001))
    # 形成矩阵
    index = 0
    for line in arrayLines:
        tmp = []
        line = line.strip() 
        tmp = re.findall(r'\d+', line)
        #添加常数1
        tmp[0:0]=[1]
        myMat[index,:] = tmp
        index += 1

    fr_label = open('train_labels.txt')
    arrayLabels = fr_label.readlines()
    
    classLabel= []
    
    for item in arrayLabels:
        classLabel.append((int)(item))
 
    return myMat,classLabel,bigNum
# 测试数据   
def testData():
    fr_data = open('test_data.txt')
    arrayLines = fr_data.readlines()
    myMat = zeros((len(arrayLines),10001))

    index = 0
    for line in arrayLines:
        line = line.strip()
        tmp = []
        line = line.strip() 
        tmp = re.findall(r'\d+', line)
        tmp[0:0]=[1]
        myMat[index,:] = tmp
        index += 1

    fr_label = open('test_labels.txt')
    arrayLabels = fr_label.readlines()
    
    classLabel= []
    # 标签
    for item in arrayLabels:
        classLabel.append((int)(item))
        
    return myMat,classLabel

# 计算sign
def sign(data):
    if data > 0:
        return 1
    else:
        return -1

# 归一化 
def autoNorm(dataSet):
    # 数据中最小值
    minVals = dataSet.min(1)
    # 数据中最大值
    maxVals = dataSet.max(1)
    ranges = maxVals - minVals
    m,n = dataSet.shape
    minVals = tile(minVals, (n,1)).T
    ranges = tile(ranges, (n,1)).T
    normDataSet = zeros(shape(dataSet))
    # 归一化数据 = 数据 -最小数 / (最大数 - 最小数)
    normDataSet = dataSet - minVals
    normDataSet = normDataSet/ranges
    return normDataSet
    
def pocket(dataMat, classLabels,mode,weights):
    m, n = shape(dataMat)
    W_pocket = weights
    if(mode == 1):
        # 用误分数据更新权重后，下一次选择数据是从误分数据往后的数据
        for num in range(1000):
            isCompleted = True
            for i in range(m):
                # 如果第i个数据可以被分好，则计算下一个数据
                if (sign(dot(dataMat[i], weights.T)) == classLabels[i]):
                    continue
                # 如果数据被分错了，就更新权重
                # W(t+1) <- W(t) + y(t) * x(i)
                else:
                    isCompleted = False
                    weights = weights + dataMat[i] * classLabels[i]
                    myLablel1 = classifyAll(dataMat, weights)
                    myLablel2 = classifyAll(dataMat, W_pocket)
                    if(Accuracy(classLabels,myLablel1) > Accuracy(classLabels,myLablel2)):
                        W_pocket = weights
            if isCompleted:
                break
    else:
        # 重新从头选数据
        for num in range(10000):
            isCompleted = True
             # 用误分数据更新权重后，下一次选择数据是从头开始的数据
            for i in range(m):
                # 如果第i个数据可以被分好，则计算下一个数据
                if (sign(dot(dataMat[i], weights.T)) == classLabels[i]):
                    continue
                # 如果数据被分错了，就更新权重
                # W(t+1) <- W(t) + y(t) * x(i)
                else:
                    isCompleted = False
                    # 更新时除以数据大小，使得新权重和旧权重相差没有这么大
                    #weights = weights + classLabels[i] * dataMat[i] / dataMat[i].sum()
                    weights = weights + classLabels[i] * dataMat[i]
                    # 用现在的权重分类训练数据，得到正确率
                    myLablel1 = classifyAll(dataMat, weights)
                    myLablel2 = classifyAll(dataMat, W_pocket)
                    # 如果用新权重正确率大于用pocket里的权重，则更新pocket里的权重
                    if(Accuracy(classLabels,myLablel1) > Accuracy(classLabels,myLablel2)):
                        W_pocket = weights
                    break
            # 假如所有数据都分好了，就不用继续更新权重了
            if isCompleted:
                break
    return W_pocket
   
# 分类
def classifyAll(dataSet, weights):
    myLablels = []
    for vector in dataSet:
        myLablels.append(sign(sum(vector * weights)))
    return myLablels

# 计算Accuracy
def Accuracy(testLabels,myLablels):
    TP = FN = FP = TN = 0
    for index in range(len(testLabels)):
        if(testLabels[index] == 1 and myLablels[index] == 1):
            TP += 1
        elif (testLabels[index] == 1 and myLablels[index] == -1):
            FN += 1
        elif (testLabels[index] == -1 and myLablels[index] == 1):
            FP += 1
        else:
            TN += 1

    return float(TP + TN) / float(TP + FP + TN + FN)
# 计算Recall
def Recall(testLabels,myLablels):
    TP = FN = FP = TN = 0
    for index in range(len(testLabels)):
        if(testLabels[index] == 1 and myLablels[index] == 1):
            TP += 1
        elif (testLabels[index] == 1 and myLablels[index] == -1):
            FN += 1
        elif (testLabels[index] == -1 and myLablels[index] == 1):
            FP += 1
        else:
            TN += 1

    return float(TP) / float(TP + FN)
# 计算Precision
def Precision(testLabels,myLablels):
    TP = FN = FP = TN = 0
    for index in range(len(testLabels)):
        if(testLabels[index] == 1 and myLablels[index] == 1):
            TP += 1
        elif (testLabels[index] == 1 and myLablels[index] == -1):
            FN += 1
        elif (testLabels[index] == -1 and myLablels[index] == 1):
            FP += 1
        else:
            TN += 1

    return float(TP) / float(TP + FP)
# 计算F1
def F1(testLabels,myLablels):
    return 2 * Precision(testLabels,myLablels) * Recall(testLabels,myLablels) / float(Precision(testLabels,myLablels) + Recall(testLabels,myLablels))
    
def main():
    trainDataSet, trainLabels,bigNum = trainData()
    testDataSet,testLabels = testData(bigNum)
    m,n = shape(trainDataSet)
    '''
    # 归一化数据
    trainDataSet = autoNorm(trainDataSet)
    testDataSet = autoNorm(testDataSet)
    '''
    line1 = []
    line2 = []
    line3 = []
    line4 = []

    for j in range(3):
        if(j == 0):
            weights = zeros((1,n))
            print ("初始化为0")
        elif(j == 1):
            weights = ones((1,n))
            print ("初始化为1")
        else:
            weights = numpy.random.randn(1,n)
            print ("随机初始化")
        for i in range(2):
            if(i == 0):
                print ("  从头选数据")
            else:
                print ("  从下一个数据开始")
            W = pocket(trainDataSet, trainLabels,i,weights)
            myLablels = classifyAll(testDataSet, W)
            print ("    口袋算法：")
            acc=Accuracy(testLabels,myLablels)
            rec=Recall(testLabels,myLablels)
            pre=Precision(testLabels,myLablels)
            f1=F1(testLabels,myLablels)
            print "Accuracy = " , acc
            print "Recall = " ,rec
            print "Precision = " , pre
            print "F1 = " , f1
            # 画图数据
            line1.append(acc)
            line2.append(rec)
            line3.append(pre)
            line4.append(f1)
    # 画图
    plt.xlim(0, 8)
    plt.plot(line1,'r',label="Accuracy")
    plt.plot(line2,'g',label="Recall")
    plt.plot(line3,'b',label="Precision")
    plt.plot(line4,'k',label="F1")
    plt.legend()# make legend
if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()