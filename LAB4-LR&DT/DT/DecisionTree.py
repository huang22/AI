from math import log
import operator
import treePlotter
import math
from math import *
from numpy import ndarray

node1 = []
node2 = []
node3 = []
 
def calEnt(dataSet):
    """
        计算样本的熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        dataLabel = featVec[-1]
        # 属于两种标签样本数
        if dataLabel not in labelCounts.keys():
            labelCounts[dataLabel] = 0
        labelCounts[dataLabel] += 1
    entroy = 0.0
    # 计算熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        entroy -= prob * log(prob, 2)
    return entroy
    
def splitDataSet(dataSet, axis, value):
    """
        按照给定特征划分数据集
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet,tree):
    """
        选择最好的数据集划分维度
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calEnt(dataSet)
    bestInfoGain = 0.0
    bestInfoGainRatio = 0.0
    bestFeature = -1
    bestGini = 999999.0
    featList = []
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        splitInfo = 0.0
        gini = 0.0
        # 根据基尼系数得到CART
        if(tree == "CART"):
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                # 这种类别比例
                prob = len(subDataSet)/float(len(dataSet))
                # 每种类别样本比例
                subProb = len(splitDataSet(subDataSet, -1, 0)) / float(len(subDataSet))
                # 计算基尼系数
                gini += prob * (1.0 - pow(subProb, 2) - pow(1 - subProb, 2))
            # 选择基尼系数小的
            if (gini < bestGini):
                bestGini = gini
                bestFeature = i
        # 根据信息增益得到ID3
        elif(tree == "ID3"):
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                # 计算熵
                newEntropy += prob * calEnt(subDataSet)
            # 信息增益
            infoGain = baseEntropy - newEntropy
            # 选择信息增益大的
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
        # 根据信息增益率得到C4.5
        else:
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet)/float(len(dataSet))
                newEntropy += prob * calEnt(subDataSet)
                # 计算数据集下关于新节点的熵
                splitInfo += -prob * log(prob, 2)
            infoGain = baseEntropy - newEntropy
            if (splitInfo == 0): 
                continue
            # 信息增益率
            infoGainRatio = infoGain / splitInfo
            # 选择信息增益率大的
            if (infoGainRatio > bestInfoGainRatio):
                bestInfoGainRatio = infoGainRatio
                bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
        采用多数判决的方法决定该子节点的分类
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    s = sorted(classCount.iteritems(), key=operator.itemgetter(1))

    return s[-1][0]  

def errCount(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sorted(classCount.iteritems(), key=operator.itemgetter(1),reverse = True)
    errorCount = 0
    # 多数投票，少的一方即为错误
    for value in classCount.values()[:-1]:
        errorCount += value
    return errorCount,len(classCount)
    
def prePurning(dataSet, dataLabel,tree):
    """
        剪枝
    """
    error = 0
    bestFeat = chooseBestFeatureToSplit(dataSet,tree)
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        splitData = splitDataSet(dataSet, bestFeat, value)
        classList = [example[-1] for example in splitData]
        errorCount,classCount = errCount(classList)
        error = error + (float)(errorCount)
    return error

def createTree(dataSet, dataLabel,tree):
    """
        递归构建决策树
    """
    flag = 0
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        if(tree == "ID3"):
            node1.append(len(classList))
        elif(tree == "C4.5"):
            node2.append(len(classList))    
        else:
            node3.append(len(classList))
        # 类别完全相同，停止划分
        return classList[0]
    
    errorCount,classCount = errCount(classList)
    beforeErr = errorCount + (2 * len(node)) * 1
    aftErr = prePurning(dataSet, dataLabel,tree)
    afterErr = aftErr + (2 * len(node) + 1) * 1
    if beforeErr >= afterErr and classCount < 3:
        flag = 1
    
    if len(dataSet[0]) == 1 or flag == 1:
        if(tree == "ID3"):
            node1.append(len(classList))
        elif(tree == "C4.5"):
            node2.append(len(classList))    
        else:
            node3.append(len(classList))
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet,tree)
    bestFeatLabel = dataLabel[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(dataLabel[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = dataLabel[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels,tree)
    #print len(allnode)
    return myTree
    

def classify(inputTree, dataLabel, testVec):
    """
    分类标签
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = dataLabel.index(firstStr)
    classLabel = 0
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], dataLabel, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def classifyAll(inputTree, dataLabel, testDataSet):
    """
    递归得到结果
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, dataLabel, testVec))
    return classLabelAll
'''
#没有离散化
def trainData(i):
    #训练集和验证集

    fr = open('train.csv')
    arrayLines = fr.readlines()
    trainMat = []
    for line in arrayLines:
        dataMat = [];
        lineArr = line.strip().split(',')
        for i in range(10):
            dataMat.append(int(lineArr[i]))
        trainMat.append(dataMat)
    trainLabel = [1,2,3,4,5,6,7,8,9]
    return trainMat,trainLabel
      
def testData():
   
    #测试集
 
    labelMat = []
    fr = open('test.csv')
    arrayLines = fr.readlines()
    myMat = []
    index = 0
    for line in arrayLines:
        dataMat = [];
        lineArr = line.strip().split(',')
        for i in range(9):
            dataMat.append(int(lineArr[i]))
        myMat.append(dataMat)
        labelMat.append(int(lineArr[9]))
        index += 1
        
    return myMat,labelMat

'''
def trainData(k):
    #训练集
    fr = open('train.csv')
    arrayLines = fr.readlines()
    trainMat = []
    myMat = zeros((len(arrayLines),10))
    index = 0
    for line in arrayLines:
        dataMat = [];
        lineArr = line.strip().split(',')
        for i in range(10):
            dataMat.append(int(lineArr[i]))
        myMat[index,:] = dataMat
        index += 1

    meanList = []
    for i in range(len(myMat[0])): 
        # 把一列的均值放进列表
        meanList.append(mean(myMat[:,i]))
        #meanList.append((max(myMat[:,i])+min(myMat[:,i]))/2.0)
        
    index = 0
    for i in range(len(myMat)):
        dataList = []
        for j in range(len(myMat[0])):
            # 假如大于均值为1
            if myMat[i][j] > meanList[j]:
                dataList.append(1)
            # 小于等于均值为0
            else:
                dataList.append(0)
        trainMat.append(dataList)
    trainLabel = [1,2,3,4,5,6,7,8,9]
    return trainMat,trainLabel,meanList
      
def testData(meanList):
        #测试集
    labelMat = []
    fr = open('test.csv')
    arrayLines = fr.readlines()
    myMat = []
    index = 0
    for line in arrayLines:
        dataMat = [];
        lineArr = line.strip().split(',')
        for i in range(9):
            if((int)(lineArr[i]) > meanList[i]):
                dataMat.append(1)
            else:
                dataMat.append(0)
        myMat.append(dataMat)
        labelMat.append(int(lineArr[9]))
        index += 1
        
    return myMat,labelMat

def Accuracy(testLabels,myLablels):
    TP = FN = FP = TN = 0
    for index in range(len(testLabels)):
        if(testLabels[index] == 1 and myLablels[index] == 1):
            TP += 1
        elif (testLabels[index] == 1 and myLablels[index] == 0):
            FN += 1
        elif (testLabels[index] == 0 and myLablels[index] == 1):
            FP += 1
        else:
            TN += 1

    return float(TP + TN) / float(TP + FP + TN + FN)

def epsion(testLabels,myLablels,tree):
    TP = FN = FP = TN = 0
    for index in range(len(testLabels)):
        if(testLabels[index] == 1 and myLablels[index] == 1):
            TP += 1
        elif (testLabels[index] == 1 and myLablels[index] == 0):
            FN += 1
        elif (testLabels[index] == 0 and myLablels[index] == 1):
            FP += 1
        else:
            TN += 1
    if(tree == "ID3"):
        node = len(node1)
    elif(tree == "C3.5"):
        node = len(node2)   
    else:
        node = len(node3)
    return float(FP + FN + node * 1) / float(TP + FP + TN + FN)
 
def main():
    trainMat,trainLabel,meanList = trainData(0)
    #trainMat,trainLabel = trainData(0)
    
    tree = "ID3"
    for i in range(3):
        if(i == 0):
            print "-------------ID3------------"
            tree = "ID3"
        elif(i == 1):
            print "-------------C4.5------------"
            tree = "C4.5"
        else:
            print "-------------CART------------"
            tree = "CART"
        labels_tmp = trainLabel[:] 
        desicionTree = createTree(trainMat, labels_tmp,tree)    
        treePlotter.createPlot(desicionTree)
        testDataSet,testLabels = testData(meanList)
        #testDataSet,testLabels = testData()
        myLabels = classifyAll(desicionTree, trainLabel, testDataSet)
        acc=Accuracy(testLabels,myLabels)
        eps = epsion(testLabels,myLabels,tree)
        print "Accuracy = " , acc
        print "Error = " , eps
    
if __name__ == '__main__':
    main()
    print "三种决策树叶子节点个数"
    print len(node1),len(node2),len(node3)