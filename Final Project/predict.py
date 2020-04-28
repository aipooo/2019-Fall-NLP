import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Encoder, Decoder, Net
from train import getDict, LoadIndexDataset
from nltk.translate.bleu_score import sentence_bleu


src_w2iDict, src_i2wDict = getDict('./index_dataset/index2word_train_source_8000.txt')
tag_w2iDict, tag_i2wDict = getDict('./index_dataset/index2word_train_target_8000.txt')
src_vocab_size = len(src_w2iDict)
tag_vocab_size = len(tag_i2wDict)

x_test = LoadIndexDataset('./index_dataset/index_test_source_1000.txt', src_i2wDict)
y_test = LoadIndexDataset('./index_dataset/index_test_target_1000.txt', src_i2wDict)
x_train = LoadIndexDataset('./index_dataset/index_train_source_8000.txt', src_i2wDict)

network = torch.load('network.pkl')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_tag_len = 100


#inputs:(src_seq_len, src_sen_len)

pred = []
for src_index in x_train[:10]:
    src_index = [src_index]
    src_index = torch.LongTensor(src_index).to(device)
    src_index = src_index.permute(1, 0)
    target_index = network.predict(src_index, tag_w2iDict['<BOS>'], max_tag_len)
    target_index = target_index.tolist()
    target_index = [int(item[0]) for item in target_index]
    target_sentence = []
    for item in target_index:
        target_sentence.append(tag_i2wDict[item])
    pred.append(target_sentence)

real = []
for sentence in y_test:
    real.append([tag_i2wDict[item] for item in sentence])
print(real)


file = open('result.txt', 'w', encoding='utf-8')
for index in range(len(pred)):
    file.write('1\n')
    for word in pred[index]:
        file.write(word)
        file.write(' ')
    file.write('\n')
    score = sentence_bleu([real[index]], pred[index])
    file.write('BLEU: ' + str(score))
    file.write('\n\n')
file.close()




