from jieba import posseg
from collections import Counter
import time
import math
import os
import codecs  # 编码转换
from sklearn.datasets.base import Bunch

stopWords = dict()
featureWords = list()  # featureWords[type][word] = P(word|type)
wholeWF = list()  # wholeWF[type][word] = 该类型的word词频
wholeDF = list()  # wholeDF[type][word] = 该类型word被引用的文档数
totalWordsCount = list()
wholeWF.append(Counter())
wholeDF.append(dict())
featureWords.append(dict())
totalWordsCount.append(0)
# 评价算法
TP = list()  # TP[type] 表示分类器识别正确，分类器认为该样本为type
FP = list()  # FP[type] 分类器识别结果错误，分类器认为该样本为type, 实际上该样本不是type
FN = list()  # FN[type] 分类器识别结果错误，分类器认为该样本不是type, 实际上该样本是type
resultMatrix = list()
correctRate = list()
recallRate = list()


# 构造停用词字典
def CreateStopWords():
    global stopWords
    with codecs.open("./res/stop_words_ch.txt", 'r', encoding='ANSI') as stopFile:  # 用ansi的编码打开文件
        for word in stopFile:
            stopWords[str(word.strip())] = 1  # 去除首尾空格


# 预处理：对文档做分词、去停用词等准备工作
def Preprocess():
    # 遍历每一个类别
    global wholeDF
    global wholeWF
    global stopWords

    countType = 6
    for curDir, subDir, files in os.walk("./res/originalTexts/6", topdown=True):  # 自顶向下扫描目录下所有子目录和文件
        if files:  # 当前目录下有文件
            # countType += 1  # 当前遍历的类型
            cutTextDir = os.path.join("./res/cutTexts", f"{countType}")  # 分词后的文件的存放目录
            cleanDir = os.path.join("./res/cleanTexts", f"{countType}")  # 移除非名词性与停用词的文件目录
            # 存放目录不存在则创建一个
            if not os.path.exists(cutTextDir):
                os.makedirs(cutTextDir)
            if not os.path.exists(cleanDir):
                os.makedirs(cleanDir)

            typeWordFrequency = Counter()
            typeDocFrequency = dict()  # typeDocFrequency[word] = word出现的文本数

            countFiles = 0  # 遍历的当前类型的文件数
            # 遍历每一个文件
            for curFileName in files:
                countFiles += 1
                writeFilesName = f"{countFiles}.txt"
                if countFiles > 10000:
                    break
                curReadPath = os.path.join(curDir, curFileName)  # 当前读取文件的完整路径
                # 读取原始文件
                with codecs.open(f"{curReadPath}", 'r', encoding='utf-8') as curReadFile:
                    oriContent = curReadFile.read()  # 分割前的原始文本内容
                    words = posseg.cut(oriContent)  # 分词并标注词性
                    curWritePath = os.path.join(cutTextDir, writeFilesName)  # 要写入的文件的地址
                    # 把分词结果写入文件
                    wordList = list()
                    with codecs.open(f"{curWritePath}", 'w', encoding='utf-8') as curWriteFile:
                        for curWord in words:
                            wordStr = str(curWord.word)
                            flagStr = str(curWord.flag)
                            writeData = f"{str(curWord.word)} {str(curWord.flag)}\n"
                            writeData.encode("utf-8")
                            curWriteFile.write(writeData)
                            # 过滤非名词和停止词
                            if 'n' in flagStr and flagStr != "nr" and wordStr not in stopWords:  # 过滤
                                wordList.append(wordStr)
                                if wordStr in typeDocFrequency:  # 更新该词被引用的文档数
                                    typeDocFrequency[wordStr] += 1
                                else:
                                    typeDocFrequency[wordStr] = 1

                    # 过滤非名词和停止词和人名
                    cleanFilePath = os.path.join(cleanDir, writeFilesName)  # 要写入的文件的地址
                    with codecs.open(f"{cleanFilePath}", 'w', encoding='utf-8') as cleanFile:
                        listSize = len(wordList)
                        for i in range(0, listSize):
                            cleanFile.write(f"{wordList[i]} ")  # 符合条件的词写入文件

                    fileWordFrequency = Counter(wordList)  # 该文件中含有的合法词
                    if countFiles <= 5000:  # 取前5000个训练
                        typeWordFrequency.update(wordList)  # 计算该类型词频
                    else:  # 后5000个测试集，记录每个集合词频最高的400个词
                        testFilePath = f"./res/TestFiles/{countType}/{writeFilesName}"
                        with codecs.open(testFilePath, 'w', encoding='utf-8') as cleanTestFile:
                            maxIndex = 400
                            if len(fileWordFrequency) < 400:
                                maxIndex = len(fileWordFrequency)
                            temp = fileWordFrequency.most_common(maxIndex)
                            for i in range(0, maxIndex):
                                cleanTestFile.write(f"{temp[i][0]}\n")
            wholeDF.append(typeDocFrequency)
            wholeWF.append(typeWordFrequency)


# 构造whole数组，统计每个类别中的词频和被引用的文档数
def ReadTrainCleanFile():
    global wholeDF  # 某个类别的干净词频数字典
    global wholeWF  # 某个类别的干净词在该类别中出现的文本数
    global stopWords

    for curType in range(1, 11):  # 遍历每个类别的文本
        typeWordFrequency = Counter()  # 统计当前的类别每个词出现的频率
        typeDocFrequency = dict()  # typeDocFrequency[word] = word出现的文本数

        cleanDir = os.path.join("./res/cleanTexts", f"{curType}")  # 移除非名词性与停用词的文件目录
        for curFile in range(1, 5001):
            fileCounter = Counter()
            cleanFilePath = os.path.join(cleanDir, f"{curFile}.txt")  # 要读入的干净词文件的地址
            with codecs.open(f"{cleanFilePath}", 'r', encoding='utf-8') as cleanFile:
                wordList = cleanFile.read().split()
                typeWordFrequency.update(wordList)  # 计算该类型词频
                fileCounter.update(wordList)
            for wordStr in fileCounter.keys():
                if wordStr in typeDocFrequency:  # 更新该词被引用的文档数
                    typeDocFrequency[wordStr] += 1
                else:
                    typeDocFrequency[wordStr] = 1
        wholeDF.append(typeDocFrequency)
        wholeWF.append(typeWordFrequency)


# 读取分词后的文件并过滤，过滤后写入clean文件
def ReadCutFile():
    for curType in range(1, 11):
        cutTextDir = os.path.join("./res/cutTexts", f"{curType}")  # 分词后的文件的存放目录
        cleanDir = os.path.join("./res/cleanTexts", f"{curType}")  # 移除非名词性与停用词的文件目录
        for curFile in range(1, 10001):
            cutFilePath = os.path.join(cutTextDir, f"{curFile}.txt")  # 文件的地址
            cleanWords = list()
            with codecs.open(f"{cutFilePath}", 'r', encoding='utf-8') as cutFile:
                wordList = cutFile.read().split('\n')  # 该文本对应的所有分词
                for i in range(0, len(wordList)):
                    curPair = wordList[i].split(' ')
                    if len(curPair) == 2:
                        wordStr = curPair[0]  # 当前遍历的词
                        flagStr = curPair[1]  # 当前遍历的词的属性
                        # 过滤非名词和人名和停止词
                        if 'n' in flagStr and flagStr != "nr" and wordStr not in stopWords:
                            cleanWords.append(wordStr)  #符合条件的词标记位干净的词

            cleanFilePath = os.path.join(cleanDir, f"{curFile}.txt")  # 要写入的文件的地址
            with codecs.open(f"{cleanFilePath}", 'w', encoding='utf-8') as cleanFile:
                listSize = len(cleanWords)
                for i in range(0, listSize):
                    cleanFile.write(f"{cleanWords[i]} ")  # 符合条件的词写入文件


# 读取干净的测试词集，提取特征词
def CleanTestFile():
    for curType in range(1, 11):
        for curFile in range(5001, 10001):
            cleanFilePath = f"./res/cleanTexts/{curType}/{curFile}.txt"
            testFilePath = f"./res/TestFiles/{curType}/{curFile}.txt"
            fileWordFrequency = Counter()

            with codecs.open(f"{cleanFilePath}", 'r', encoding='utf-8') as cleanFile:  # read
                wordList = cleanFile.read().split()
                fileWordFrequency.update(wordList)  # 计算该类型词频

            with codecs.open(testFilePath, 'w', encoding='utf-8') as cleanTestFile:  # write
                maxIndex = 400
                if len(fileWordFrequency) < 400:
                    maxIndex = len(fileWordFrequency)
                temp = fileWordFrequency.most_common(maxIndex)
                for i in range(0, maxIndex):
                    cleanTestFile.write(f"{temp[i][0]}\n")


# 提取每个类别的训练集中的特征词,最多取400个
def ExtractFeature(mode):
    global wholeWF
    for curType in range(1, 11):  # 共10个分类
        maxIndex = 400
        if len(wholeWF[curType]) < 400:
            maxIndex = len(wholeWF[curType])

        if mode == "CHI":  # 选卡方值最大的做特征词
            wordCHI = ComputeCHI(curType)
            sortWordChi = sorted(wordCHI.items(), key=lambda x: x[1], reverse=True)[:maxIndex]  # 按chi的值降序排列
            ComputeWordP(curType, sortWordChi, maxIndex)
        elif mode == "TF/IDF":
            pass
        elif mode == "DF":  # 选词频最高的做特征词
            ComputeWordP(curType, wholeWF[curType].most_common(maxIndex), maxIndex)


# 计算特征词在某个类别中的似然度P(word|type) ， 采用拉普拉斯进行平滑处理
def ComputeWordP(curType, wordList, maxIndex):
    global featureWords
    global wholeWF
    feature = dict()
    totalWords = maxIndex  # 本类型文档中特征词的总个数（特征词最多取400个）, 采用拉普拉斯平滑，分母加上词数
    for i in range(0, maxIndex):
        curWord = wordList[i][0]  # wordList[i]是一个二元元组（word，value）
        totalWords += wholeWF[curType][curWord]
    totalWordsCount.append(totalWords)

    # 采用拉普拉斯平滑，公式如下：
    # P(word|type) = （word在type的所有文档中出现的次数 + 1）/ （本类型文档中特征词的总个数+特征词数）
    featureFilePath = f"./res/featureWords/{curType}.txt"
    with codecs.open(f"{featureFilePath}", 'w', encoding='utf-8') as featureFile:
        featureFile.write(f"{totalWords}\n")
        for i in range(0, maxIndex):
            curWord = wordList[i][0]
            p = float(wholeWF[curType][curWord] + 1) / float(totalWords)  # 计算P(word|type)
            feature[curWord] = p
            featureFile.write(f"{str(curWord)} {p}\n")
    featureWords.append(feature)


def ComputeCHI(curType):
    # 𝑋2(word, type) = 𝑁 × (𝐴𝐷 − 𝐵𝐶)^2 /(𝐴 + 𝐶)(𝐵 + 𝐷)(𝐴 + 𝐵)(𝐶 + 𝐷)
    # N = 10 * 5000, 训练集文本总数
    # A 为训练集中类别type包含词语word的文档数, C 为类别type中不含word的文档数，A+C为5000
    # B 为训练集中类别不为type包含词语word的文档数, C 为类别不为type中不含word的文档数，B+D = 5000 * 9
    # 所以卡方计算可以简化成 (𝐴𝐷 − 𝐵𝐶)/(𝐴 + 𝐵)(𝐶 + 𝐷)
    global wholeDF
    wordCHI = dict()
    for word, A in wholeDF[curType].items():  # 遍历当前类型的所有训练文档中包含的合法词
        C = 5000 - A
        B = ComputeB(word, curType)
        D = 45000 - B
        wordCHI[word] = (A * D - B * C) * (A * D - B * C) / float(A + B) / float(C + D)
    return wordCHI


# B 为训练集中类别不为type包含词语word的文档数
def ComputeB(word, exceptType):
    global wholeDF
    countB = 0
    for curType in range(1, 11):
        if curType == exceptType:
            continue
        else:
            if word in wholeDF[curType]:
                countB += wholeDF[curType][word]
    return countB


def Init():
    global TP  # TP[type] 表示分类器识别正确，分类器认为该样本为type
    global FP  # FP[type] 分类器识别结果错误，分类器认为该样本为type, 实际上该样本不是type
    global FN  # FN[type] 分类器识别结果错误，分类器认为该样本不是type, 实际上该样本是type
    global correctRate
    global recallRate
    for curType in range(0, 11):
        TP.append(0)  # TP[curType] = 0
        FP.append(0)
        FN.append(0)
        correctRate.append(0.0)
        recallRate.append(0.0)
        resultMatrix.append(list())
        for i in range(0, 11):
            resultMatrix[curType].append(0)


def NBClassify():
    global TP  # TP[type] 表示分类器识别正确，分类器认为该样本为type
    global FP  # FP[type] 分类器识别结果错误，分类器认为该样本为type, 实际上该样本不是type
    global FN  # FN[type] 分类器识别结果错误，分类器认为该样本不是type, 实际上该样本是type

    # 每一个类别有5000个测试集
    for curTestType in range(1, 11):
        for curTestFile in range(5001, 10001):
            testFilePath = f"./res/TestFiles/{curTestType}/{curTestFile}.txt"
            testWordList = GetTestWords(testFilePath)  # 获取测试集的合法词列表

            # 测试每一种类别的概率
            resultType = ComputeNBType(testWordList)
            resultMatrix[curTestType][resultType] += 1
            # 将结果与正确类别进行比较
            if resultType == curTestType:  # 成功
                TP[resultType] += 1
            else:  # 失败
                FP[resultType] += 1
                FN[curTestType] += 1


# 评估分类器并打印结果
def EvaluateClassifier():
    # 正确率 = TP / (TP + FP)
    # 召回率 = TP / (TP + FN)
    global TP  # TP[type] 表示分类器识别正确，分类器认为该样本为type
    global FP  # FP[type] 分类器识别结果错误，分类器认为该样本为type, 实际上该样本不是type
    global FN  # FN[type] 分类器识别结果错误，分类器认为该样本不是type, 实际上该样本是type
    global correctRate
    global recallRate

    totalCorrect = 0.0
    totalRecall = 0.0

    # 计算每类正确率、召回率
    for curType in range(1, 11):
        correctRate[curType] = float(TP[curType]) / float(TP[curType] + FP[curType])
        recallRate[curType] = float(TP[curType]) / float(TP[curType] + FN[curType])
        print(f"Type {curType}: 正确率：{correctRate[curType]}  召回率：{recallRate[curType]}\n")
        totalCorrect += correctRate[curType]
        totalRecall += recallRate[curType]
    # 计算总体正确率、召回率
    totalCorrect /= 10
    totalRecall /= 10
    print(f"Total: 正确率：{totalCorrect}  召回率：{totalRecall}\n")

    print(f"{str(1): >12}{str(2): >6}{str(3): >6}{str(4): >6}{str(5): >6}{str(6): >6}{str(7): >6}{str(8): >6}{str(9): >6}{str(10): >6}\n")
    for i in range(1, 11):
        outputStr = f"{str(i).rjust(6, ' ')}"
        for j in range(1, 11):
            outputStr += f"{str(resultMatrix[i][j]).rjust(6, ' ')}"
        print(f"{outputStr}\n")


def GetTestWords(filePath):
    wordList = list()
    with codecs.open(filePath, 'r', encoding='utf-8') as curReadFile:
        for word in curReadFile:
            curWord = str(word.strip())
            wordList.append(curWord)
    return wordList


def ReadFeature():
    global featureWords
    global totalWordsCount
    for curType in range(1, 11):
        featureWords.append(dict())
        featureFilePath = f"./res/featureWords/{curType}.txt"
        with codecs.open(f"{featureFilePath}", 'r', encoding='utf-8') as featureFile:
            wordList = featureFile.read().split('\n')
            totalWordsCount.append(int(wordList[0]))
            for i in range(1, len(wordList)):
                curPair = wordList[i].split(' ')
                if len(curPair) == 2:
                    featureWords[curType][curPair[0]] = float(curPair[1])


# 利用朴素贝叶斯进行分类
def ComputeNBType(testWordsList):
    # 假设每种类别文档的概率都相同，P(type) = 1/10
    # NB(type) = P(testFile|type)*P(type)
    # 记testFile里的特征词分别为w1,w2...wn，则
    # P(testFile|type) = P(w1|type)*P(w2|type)*...*P(wn|type), 实际计算中用log的加法代替乘法

    global totalWordsCount  # totalWordsCount[type] = type类型对应的总词数
    global featureWords  # featureWords[type]=type类型对应的特征词
    resultType = 1
    maxPossibility = -10000000.0
    testSize = len(testWordsList)
    for curTrainType in range(1, 11):  # 测试每一种类型的NB概率
        trainTypeP = 0.0
        for i in range(0, testSize):  # 遍历testFile里的特征词
            word = testWordsList[i]
            if word in featureWords[curTrainType]:  # 如果测试集的特征词在类别特征词里
                trainTypeP += math.log2(featureWords[curTrainType][word])
            else:  # 不在类别特征词里，即该词在这类文档中频数为0
                # 根据拉普拉斯平滑， P(word|type) = (0 + 1) / totalWordsCount[type]
                trainTypeP -= math.log2(totalWordsCount[curTrainType])  # math.log2(1) = 0，省略
        if trainTypeP > maxPossibility:  # 得到最大的值的type即为分类结果
            maxPossibility = trainTypeP
            resultType = curTrainType
    return resultType


def SVM():
    pass


if __name__ == '__main__':
    Init()
    CreateStopWords()
    # Preprocess()  # 第一次处理数据集用这个

    # ReadTrainCleanFile()  # 构造whole数组
    # ExtractFeature("CHI")
    ReadFeature()
    startClassify = time.time()
    NBClassify()
    endClassify = time.time()
    print(f"测试时间：{endClassify - startClassify}")
    EvaluateClassifier()
