from numpy import *
import numpy
import matplotlib.pyplot as plt
import pylab as pl
import time

def trainData(i):
    '''
       训练集并采用交叉验证（10 folds）
    '''
    trainLabel = []
    valiLabel= []
    fr = open('train.csv')
    arrayLines = fr.readlines()
    # 十折cross validation
    trainMat = zeros((len(arrayLines) - len(arrayLines)/10,10))
    valiMat = zeros((len(arrayLines)/10,10))
    index = 0
    trainIndex = 0
    valiIndex = 0
    for line in arrayLines:
        dataMat = [];
        lineArr = line.strip().split(',')
        for i in range(9):
            dataMat.append(float(lineArr[i]))
        #添加常数1.0
        dataMat[0:0]=[1.0]
        # 训练集
        if(index < i or index >= i + len(arrayLines)/10):
            trainMat[trainIndex,:] = dataMat
            trainLabel.append((float)(lineArr[9]))
            trainIndex += 1
        # 验证集
        else:
            valiMat[valiIndex,:] = dataMat
            valiLabel.append((float)(lineArr[9]))
            valiIndex += 1
        index += 1
    #输出训练集及其标签，验证集及其标签
    return trainMat,trainLabel,valiMat,valiLabel
   
def testData():
    '''
        测试集
    '''      
    labelMat = []
    fr = open('test.csv')
    arrayLines = fr.readlines()
    myMat = zeros((len(arrayLines),10))
    index = 0
    for line in arrayLines:
        dataMat = [];
        lineArr = line.strip().split(',')
        for i in range(9):
            dataMat.append(float(lineArr[i]))
        #添加常数1.0
        dataMat[0:0]=[1.0]
        myMat[index,:] = dataMat
        labelMat.append(float(lineArr[9]))
        index += 1
        
    return myMat,labelMat

def sigmoid(inX):
    '''
        sigmoid函数
    '''
    return (1.0/(1.0+exp(-inX.sum(axis = 1))))

def autoNorm(dataSet):
    '''
        归一化
    '''
    # 数据中最小值
    meanVals = dataSet.mean(axis=1)
    # 数据中最大值
    stdVals = dataSet.std(axis=1)
    m,n = dataSet.shape
    meanVals = tile(meanVals, (n,1)).T
    stdVals = tile(stdVals, (n,1)).T
    normDataSet = zeros(shape(dataSet))
    # 归一化数据 = 数据 -最小数 / (最大数 - 最小数)
    normDataSet = dataSet - meanVals
    normDataSet = normDataSet/stdVals
    return normDataSet

def trainLog(dataMatIn, classLabels,opts,weights):
    '''
        LR训练
    '''
    dataMatrix = mat(dataMatIn)          
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    eta = opts['eta']
    maxCycles = opts['maxCycles'] 
    lam = 0.01
    

    for k in range(maxCycles):           
        # 梯度下降算法
        eta = 4.0 / (1.0 + k) + 0.01
        if opts['optimizeType'] == 'gradDescent': 
            output =(sigmoid(dataMatrix * weights.T))
            error = (output - labelMat).T * dataMatrix
            if abs(sum(error)) < 0.001:
                break
            else:
                weights = weights - eta * error 
        # 对每个样本更新权值
        elif opts['optimizeType'] == 'plaGradDescent': 
            for i in range(m):
                eta = 4.0 / (1.0 + k + i) + 0.01
                output = sigmoid(dataMatrix[i, :] * weights.T)  
                error = output - labelMat[i, 0]
                weights = weights - eta * error * dataMatrix[i, :]
            output =(sigmoid(dataMatrix * weights.T))
            error = (output - labelMat).T * dataMatrix
            if abs(sum(error)) < 0.001:
                break
        # 随机下降算法
        elif opts['optimizeType'] == 'stocGradDescent': 
            dataIndex = range(m)
            for i in range(m):  
                eta = 4.0 / (1.0 + k + i) + 0.01 
                # 随机选择一个样本
                randIndex = int(random.uniform(0, len(dataIndex)))
                output = sigmoid(dataMatrix[randIndex, :] * weights.T)  
                error = output - labelMat[randIndex, 0]
                weights = weights - eta * error * dataMatrix[randIndex, :]
                # 从列表删除这个更新过得样本 
                del(dataIndex[randIndex])  
            output =(sigmoid(dataMatrix * weights.T))
            error = (output - labelMat).T * dataMatrix
            if abs(sum(error)) < 0.001:
                break
    return weights,abs(sum(error))

def classifyAll(dataSet, weights):
    '''
        得到样本的标签
    '''
    myLablels = []
    for vector in dataSet:
        for j in range(len(dataSet)):
            prob = (1.0/(1+exp(-sum(dataSet[j]*weights.T))))
            if prob > 0.5:
                myLablels.append(1)
            else:
                myLablels.append(0)
    return myLablels

def Accuracy(testLabels,myLablels):
    '''
        计算准确率
    '''
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

def Recall(testLabels,myLablels):
    '''
        计算召回率
    '''
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

    return float(TP) / float(TP + FN)
    
def Precision(testLabels,myLablels):
    '''
        计算精确率
    '''
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

    return float(TP) / float(TP + FP)
    
def F1(testLabels,myLablels):
    '''
        计算F1
    '''
    return 2 * Precision(testLabels,myLablels) * Recall(testLabels,myLablels) / float(Precision(testLabels,myLablels) + Recall(testLabels,myLablels))


def main():
    testDataSet,testLabels = testData()
    W = ones((1,10))
    m,n = shape(testDataSet)
    weights = zeros((1,n))
    
    # 学习次数固定,找最合适学习率
    opts = {'eta': 0.01, 'maxCycles': 100, 'optimizeType': 'gradDescent'}
    #print "-------------find opts-------------"
    line1 = []
    line2 = []
    line3 = []
    line4 = []
    best_error = 1000
    
    # 学习率从0.00001 到 10000 一共九级变化
    for i in range(10):
        tempOpts = {'eta': 0.00001*pow(10,i), 'maxCycles': 100, 'optimizeType': 'gradDescent'}
        trainMat,trainLabel,valiMat,valiLabel = trainData(i)
        weights,error = trainLog(trainMat, trainLabel,opts,weights)
        myLablels = classifyAll(valiMat, weights)
        # 假如指标更好，则为更好的学习率
        if best_error > error:
            best_error = error
            opts = tempOpts
    print opts
    
    for i in range(3):
        best_acc = 0.01
        best_rec = 0.01
        best_pre = 0.01
        best_f1 = 0.01
        best_error = 1000
        # 不同初始化权重
        if(i == 0):
            print "------------初始化为0------------"
            weights = zeros((1,n))
        elif(i == 1):
            print "------------初始化为1------------"
            weights = ones((1,n))
        else:
            print "------------初始化为随机数------------"
            weights = numpy.random.randn(1,n)
        '''
        #没有十折交叉验证的初始算法
        trainMat,trainLabel,valiMat,valiLabel = trainData(1)
        weights = trainLog(trainMat, trainLabel,opts,weights)
        myLablel = classifyAll(testDataSet, weights)
        acc=Accuracy(testLabels,myLablel)
        rec=Recall(testLabels,myLablel)
        pre=Precision(testLabels,myLablel)
        f1=F1(testLabels,myLablel)
        print "Accuracy = " , acc
        print "Recall = " ,rec
        print "Precision = " , pre
        print "F1 = " , f1
        '''
        for j in range(3):
            if j == 0:
                #print "-------------gradDescent-------------"
                for j1 in range(10):
                    trainMat,trainLabel,valiMat,valiLabel = trainData(j1)
                    weights,error = trainLog(trainMat, trainLabel,opts,weights)
                    myLablel = classifyAll(valiMat, weights)
                    acc=Accuracy(valiLabel,myLablel)
                    rec=Recall(testLabels,myLablel)
                    pre=Precision(testLabels,myLablel)
                    f1=F1(testLabels,myLablel)
                    '''
                    if acc >= best_acc:
                        W = weights
                        best_acc = acc
                        best_rec = rec
                        best_pre = pre
                        best_f1 = f1
                    '''
                    if best_error > error:
                        best_error = error
                        W = weights
            elif j == 1:
                print "-------------plaGradDescent-------------"
                opts['optimizeType'] = 'plaGradDescent'
                for j2 in range(5):
                    trainMat,trainLabel,valiMat,valiLabel = trainData(j2)
                    weights,error = trainLog(trainMat, trainLabel,opts,W)
                    myLablel = classifyAll(valiMat, weights)
                    acc=Accuracy(valiLabel,myLablel)
                    rec=Recall(testLabels,myLablel)
                    pre=Precision(testLabels,myLablel)
                    f1=F1(testLabels,myLablel)
                    '''
                    if acc >= best_acc:
                        W = weights
                        best_acc = acc
                        best_rec = rec
                        best_pre = pre
                        best_f1 = f1
                    '''
                    if best_error > error:
                        best_error = error
                        W = weights
            else:
                print "-------------stocGradDescent:-------------"
                opts['optimizeType'] = 'stocGradDescent'
                for j3 in range(5):
                    trainMat,trainLabel,valiMat,valiLabel = trainData(j3)
                    weights,error = trainLog(trainMat, trainLabel,opts,W)
                    myLablel = classifyAll(valiMat, weights)
                    acc=Accuracy(valiLabel,myLablel)
                    rec=Recall(testLabels,myLablel)
                    pre=Precision(testLabels,myLablel)
                    f1=F1(testLabels,myLablel)
                    '''
                    if acc >= best_acc:
                        W = weights
                        best_acc = acc
                        best_rec = rec
                        best_pre = pre
                        best_f1 = f1 
                    '''
                    if best_error > error:
                        best_error = error
                        W = weights
            myLablels = classifyAll(testDataSet, W)
            # 四种指标
            acc=Accuracy(testLabels,myLablels)
            rec=Recall(testLabels,myLablels)
            pre=Precision(testLabels,myLablels)
            f1=F1(testLabels,myLablels)
            print "Accuracy = " , acc
            print "Recall = " ,rec
            print "Precision = " , pre
            print "F1 = " , f1
         # 画图   
        line1.append(acc)
        line2.append(rec)
        line3.append(pre)
        line4.append(f1)
    plt.figure(i+1)    
    plt.xlim(0, 4)
    plt.plot(line1,'r',label="Accuracy")
    plt.plot(line2,'g',label="Recall")
    plt.plot(line3,'b',label="Precision")
    plt.plot(line4,'k',label="F1")
    plt.legend()
    

if __name__ == '__main__':
    main()
        