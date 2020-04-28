from sklearn.feature_extraction.text import CountVectorizer
import random
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


def getTestSentenceList(filename) :
    '''
    使用re.split()处理测试数据，得到的结果以list方式存储
    截取[MASK]的前面部分
    list的每一个元素为一条句子
    '''
    TestSentenceList = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('[MASK]')[0]
            TestSentenceList.append(line)
    return TestSentenceList
            

def Sentence2Words(SentenceList, StopWordsList):
    '''
    使用pkuseg分词工具对句子进行分词
    函数输入为句子列表，输出为单词列表
    '''
    WordList = []
    for sentence in SentenceList:
        sentence = ''.join(re.findall(r'[\u4e00-\u9fa5]', sentence)) 
        sentence = sentence.strip()
        tmpWordList = seg.cut(sentence)
        WordList.append(['<START>']+[word for word in tmpWordList if word!= ' ' and word not in StopWordsList])
    return WordList



def getNews(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.rstrip() != '':
                data.append(line.rstrip())
    return data

def getAllNews():
    data = []
    for index in range(1, 1001):
        filename = str(index) + '.txt'
        data += getNews(filename)
    return data

def getWordList(filename):
    word_list = ['<START>', '<END>']
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            word_list.append(line[1])
    return word_list



def predictWord(data, prestr, word_list):
    '''
    本函数对隐藏词进行预测
    data为语料库，prestr为隐藏词前的词组，word_list为单词列表
    '''
    #gram的值依据prestr的词数判定
    gram = len(prestr.split(' ')) + 1
    #利用CounterVectorizer将data拆分成gram个单词组成的单词元组
    vec = CountVectorizer(min_df=1, ngram_range=(gram, gram))
    vec_fit = vec.fit_transform(data)
    #计算各个单词元组出现的频数，构造字典
    vocab = vec.vocabulary_
    vocabDict = {k:0 for k in vocab.values()}
    for i in vec_fit:
        for j in i.indices:
            vocabDict[j] += i[0, j]

    max_num = 0
    #遍历单词列表里的所有词，构造所有的以prestr开头的元组
    #对max_num和chosen_word进行更新，找到频数最大的元组
    for item in word_list:
        s = prestr + ' ' + item
        if vocab.get(s) is None:
            continue
        if vocabDict[vocab[s]] > max_num:
            max_num = vocabDict[vocab[s]]
            chosen_word = item
    #如果max_num为0，说明以prestr开头的元组不存在与语料库中
    if max_num == 0:
        #如果gram大于2，则去除prestr开头的一个单词，进行递归
        if gram > 2:
            prestr = ' '.join(prestr.split(' ')[1:])
            return predictWord(data, prestr, word_list)
        #否则直接返回null(也可以随机挑选一个单词返回)
        else:
            return 'null'
            #return random.choice(word_list)
    #返回预测的单词chosen_word
    return chosen_word
    
        
            


if __name__ == '__main__':
    data = getAllNews()
    for item in data:
        item = '<START> ' + item + '<END>'
    word_list = getWordList('词表.txt')
    test_data = getTestSentenceList('questions.txt')
    StopWordsList = getStopWordsList()
    test_data = Sentence2Words(test_data, StopWordsList)

    predict_result = []
    for sentence in test_data:
        prestr = ' '.join(sentence[-2:])
        print(sentence)
        word = predictWord(data, prestr, word_list)
        predict_result.append(word)
        print(word)

    real_result = []
    with open('answer.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            real_result.append(line.strip())
    
    right_num = 0
    for index in range(len(real_result)):
        if real_result[index] == predict_result[index]:
            right_num += 1
            print(index)
    accuracy = right_num / len(real_result)
    
    with open('ngram_predict.txt', 'w', encoding='utf-8') as f:
        for item in predict_result:
            f.write(item)
            f.write('\n')
    
    
    



