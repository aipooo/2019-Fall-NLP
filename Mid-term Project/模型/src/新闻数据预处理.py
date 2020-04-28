import numpy as np
import re
import pkuseg

seg = pkuseg.pkuseg()


def getStopWordsList():
    '''
    创建停用词列表
    '''
    stopwords = []
    with open('ChineseStopWords.txt', 'r', encoding='gbk') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


def getSentenceList(filename) :
    '''
    使用re.split()处理新闻数据，得到的结果以list方式存储
    list的每一个元素为一条句子
    '''
    SentenceList = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            tmpSentenceList = re.split('\.|。|\!|！|\?|？', line)
            for sentence in tmpSentenceList:
                sentence = sentence.strip()
                if len(sentence)!=0:
                    SentenceList.append(sentence)
    return SentenceList
            

def Sentence2Words(SentenceList, StopWordsList):
    '''
    使用pkuseg分词工具对句子进行分词
    函数输入为句子列表，输出为单词列表
    '''
    WordList = []
    for sentence in SentenceList:
        sentence =  ''.join(re.findall(r'[\u4e00-\u9fa5]', sentence)) 
        sentence = sentence.strip()
        tmpWordList = seg.cut(sentence)
        WordList.append([word for word in tmpWordList if word!= ' ' and word not in StopWordsList])
    return WordList


def ProcessOneFile(filename, StopWordsList):
    '''
    对单个新闻进行数据处理，将新闻内容按照要求处理后存为txt文件
    文件的每行是新闻原文中的一个句子分词后的结果，以空格分隔，与原始新闻内容相对应
    返回该新闻的单词列表
    '''
    SentenceList = getSentenceList(filename)
    WordList = Sentence2Words(SentenceList, StopWordsList)
    file_path = '预处理数据/' + filename
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in WordList:
            f.write(' '.join(line))
            f.write('\n')
    return WordList

def ProcessAllFile():
    '''
    对所有新闻数据进行处理
    返回总的单词列表
    '''
    StopWordsList = getStopWordsList()
    AllWordList = []
    for index in range(1, 1001):
        filename = str(index) + '.txt'
        WordList = ProcessOneFile(filename, StopWordsList)
        for item in WordList:
            for word in item:
                if word not in AllWordList:
                    AllWordList.append(word)
    return AllWordList
    

def Word2Id(WordList):
    '''
    函数根据WordList返回一个字典，key为Data里的单词，将单词映射为一个特定Id
    创建文本文件储存预处理后构建的词表，文件每行为用空格隔开的序号以及单词
    '''
    idDict = {}
    id = 1
    with open('预处理数据/词表.txt', 'w', encoding='utf-8') as f:
        for word in WordList:
            if idDict.get(word) is None:
                idDict[word] = id
                line = str(id) + ' ' + word + '\n'
                f.write(line)
                id += 1              
    return idDict
        

if __name__ == '__main__':
    WordList = ProcessAllFile()
    Word2Id(WordList)
    