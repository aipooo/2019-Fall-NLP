import numpy as np
import torch
from torch import nn, optim
import torch.autograd as autograd # torch中自动计算梯度模块
from torch.autograd import Variable
import random
import re
import pkuseg

seg = pkuseg.pkuseg()

THRESHOLD = 20


def getAllSentences():
    '''
    获取所有的句子
    '''
    sentence_list = []
    for index in range(1, 1001):
        filename = str(index) + '.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line != '':
                    sentence_list.append(line.split(' '))
    return sentence_list


def getWordDict():
    '''
    获取词语并完成到id的映射，得到字典
    '''
    idDict = {}
    with open('词表.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        count = len(lines)
        for line in lines:
            line = line.strip().split(' ')
            idDict[line[1]] = int(line[0])
    idDict['<START>'] = 0
    return idDict


def Convert2id(Data, idDict):
    '''
    将词语映射为id存储
    '''
    idData = []
    for row in range(len(Data)):
        tmp = []
        for col in range(len(Data[row])):
            if idDict.get(Data[row][col]) is None:
                tmp.append(0)
            else:
                tmp.append(idDict[Data[row][col]])
        idData.append(tmp)
    return idData


def Padding(Data, THRESHOLD):
    newData = []
    for sentence in Data:
        if len(sentence) >= THRESHOLD:
            newData.append(sentence[:THRESHOLD])
        else:
            newData.append([0] * (THRESHOLD-len(sentence)) + sentence)
    return newData





class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.layer1 = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.layer2 = nn.LSTM(self.embedding_dim, self.hidden_dim)
        self.layer3 = nn.Linear(self.hidden_dim, 32)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(32, self.vocab_size)
    
    
    def forward(self, inputs):
        embeds = self.layer1((inputs))
        lstm_out, _ = self.layer2(embeds.view(-1, 1, self.embedding_dim))
        output = self.layer3(lstm_out.view(-1, self.hidden_dim))
        output = self.layer4(output.view(-1, 32)) 
        output = self.layer5(output)
        return output

        


slist = getAllSentences()
idDict = getWordDict()
VOCAB_SIZE = len(idDict.keys())
idData = Convert2id(slist, idDict)
train = Padding(idData, THRESHOLD)
train = np.array(train)


BATCH_SIZE = 256
EPOCH = 30
EMBEDDING_DIM = 50

model = LSTM(VOCAB_SIZE, EMBEDDING_DIM, 128)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
model.cuda()

for epoch in range(EPOCH):
    print('*********************************')
    print('epoch: ', epoch+1, 'of', EPOCH)
    running_loss = 0.0
    running_acc = 0.0
    i = 0
    while i * BATCH_SIZE < len(train):
        if (i+1)*BATCH_SIZE < len(train):
            data = train[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        else:
            data = train[i*BATCH_SIZE :]
        inputs = Variable(torch.LongTensor(data[:,:-1])).cuda()
        target = Variable(torch.LongTensor(data[:,1:])).cuda()
        target = target.view(-1)
        # forward
        output = model(inputs)
        loss = criterion(output, target)
        running_loss += loss.data.item()
        _, pred = torch.max(output, 1)  
        num_correct = (pred == target).sum()
        running_acc += num_correct.item()
        # backward
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        i += 1
    print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, EPOCH, running_loss/len(train),
        running_acc/(len(train)*(THRESHOLD-1))))

torch.save(model, 'lstm-model.pkl')


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
        WordList.append([word for word in tmpWordList if word!= ' ' and word not in StopWordsList])
    return WordList

test_data = getTestSentenceList('questions.txt')
StopWordsList = getStopWordsList()
test_data = Sentence2Words(test_data, StopWordsList)
test_data_id = Convert2id(test_data, idDict)
test = Padding(test_data_id, THRESHOLD)

reverse_idDict = {v:k for k,v in idDict.items()}

predict_result = []
for sentence in test:
    word = Variable(torch.LongTensor([sentence[:-1]])).cuda()
    output= model(word)
    _, pred = torch.max(output, 1)
    pred = pred.cpu().numpy()[-1]
    predict_result.append(reverse_idDict[pred])

real_result = []
with open('answer.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        real_result.append(line.strip())
    
right_num = 0
for index in range(len(real_result)):
    if real_result[index] == predict_result[index]:
        right_num += 1
accuracy = right_num / len(real_result)
    
with open('lstm_predict.txt', 'w', encoding='utf-8') as f:
    for item in predict_result:
        f.write(item)
        f.write('\n')

