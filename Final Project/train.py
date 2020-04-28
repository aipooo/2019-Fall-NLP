import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Encoder, Decoder, Net

def getDict(filename):
    file = open(filename, 'r', encoding='utf-8')
    lines = file.readlines()
    w2iDict = {}
    i2wDict = {}
    for line in lines:
        line = line.strip('\n')
        line = line.split()
        i2wDict[int(line[0])] = line[1]
        w2iDict[line[1]] = int(line[0])
    return w2iDict, i2wDict

def LoadIndexDataset(filename, w2iDict):
    file = open(filename, 'r', encoding='utf-8')
    lines = file.readlines()
    data = []
    for line in lines:
        line = line.strip()
        sentence = line.split()
        data.append([int(word) for word in sentence])
    return data

def padData(data, w2iDict):
    max_len = 0
    for line in data:
        if len(line) > max_len:
            max_len = len(line)
    new_data = []
    for line in data:
        new_data.append(line + [w2iDict['<PAD>']]*(max_len-len(line)))
    return new_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ ==  '__main__':
    src_w2iDict, src_i2wDict = getDict('./index_dataset/index2word_train_source_8000.txt')
    tag_w2iDict, tag_i2wDict = getDict('./index_dataset/index2word_train_target_8000.txt')
    src_vocab_size = len(src_w2iDict)
    tag_vocab_size = len(tag_i2wDict)
    
    x_train = LoadIndexDataset('./index_dataset/index_train_source_8000.txt', src_i2wDict)
    y_train = LoadIndexDataset('./index_dataset/index_train_target_8000.txt', src_i2wDict)
    x_train = x_train[:100]
    y_train = y_train[:100]
    
    hidden_dim = 256
    BATCH_SIZE = 1
    EPOCH_NUM = 10
    embed_dim = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim)
    decoder = Decoder(tag_vocab_size, embed_dim, hidden_dim)
    network = Net(encoder, decoder, device, teacher_forcing_ratio=0.5)
    
    loss_fn = nn.CrossEntropyLoss()         #使用交叉熵损失函数
    optimizer = torch.optim.Adam(network.parameters())  #使用Adam优化器
    
    for epoch in range(EPOCH_NUM):
        print('*********************************')
        print('epoch: ', epoch+1, 'of', EPOCH_NUM)
        i = 0
        while i * BATCH_SIZE < len(x_train):
            if (i+1)*BATCH_SIZE < len(x_train):
                inputs = x_train[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                target = y_train[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            else:
                inputs = x_train[i*BATCH_SIZE :]
                target = y_train[i*BATCH_SIZE :]
            inputs = padData(inputs, src_w2iDict)
            target = padData(target, tag_w2iDict)
            inputs = torch.LongTensor(inputs).to(device)
            #inputs:(batch_size, src_len)
            target = torch.LongTensor(target).to(device)
            #target:(batch_size, tag_len)
            inputs = inputs.permute(1, 0)
            target = target.permute(1, 0)
            
            outputs = network(inputs, target)
            #outpus:(tag_len, batch_size, vocab_size)
            loss = loss_fn(outputs.reshape(-1, outputs.size(2)), target.reshape(-1))
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            print(loss)
    
    torch.save(network, 'network.pkl')
    
    





