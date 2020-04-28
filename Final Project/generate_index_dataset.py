import jieba

def save_index_dict(filename, word_index_dict):
    index2word_filename = './index_dataset/index2word_' + filename
    index2word_file = open(index2word_filename, 'w', encoding='utf-8')
    for item in word_index_dict:
        index2word_file.write(str(word_index_dict[item]) + ' ' + item + '\n')
    index2word_file.close()
        

def train_word2index(filename):
    '''
    对训练集进行分词处理，同时构建训练集词表，将训练集中的词映射到对应的index
    返回得到的词表
    '''
    rfilename = './final_project_dataset/dataset_10000/' + filename
    wfilename = './index_dataset/index_' + filename
    rfile = open(rfilename, 'r', encoding='utf-8')
    wfile = open(wfilename, 'w', encoding='utf-8')
    lines = rfile.readlines()
    count = 0
    #初始化词表为空
    word_index_dict = {}
    for line in lines:
        line = line.strip().lower() #转换成小写
        seg_list = jieba.cut(line)  #利用jieba分词工具进行分词
        #在词列表的句首和句尾分别加上<BOS>和<EOS>
        word_list = ['<BOS>'] + list(seg_list) + ['<EOS>']
        #如果词不在词表中，则更新词表
        for word in word_list:
            if len(word.strip()) == 0:
                continue
            if word not in word_index_dict:
                word_index_dict[word] = count
                count += 1
            #利用此表将原训练集映射到对应index，写入文件
            wfile.write(str(word_index_dict[word]) + ' ')
        wfile.write('\n')
    #将<PAD>添加进词典
    word_index_dict['<PAD>'] = count
    rfile.close()
    wfile.close()
    #将训练集中到的词映射到对应index，保存index数据和词表
    save_index_dict(filename, word_index_dict)
    return word_index_dict

def test_word2index(filename, word_index_dict):
    rfilename = './final_project_dataset/dataset_10000/' + filename
    wfilename = './index_dataset/index_' + filename
    rfile = open(rfilename, 'r', encoding='utf-8')
    wfile = open(wfilename, 'w', encoding='utf-8')
    lines = rfile.readlines()
    for line in lines:
        line = line.strip().lower()
        seg_list = jieba.cut(line)
        word_list = ['<BOS>'] + list(seg_list) + ['<EOS>']
        for word in word_list:
            if word in word_index_dict:
                wfile.write(str(word_index_dict[word]) + ' ')
            else:
                wfile.write(str(word_index_dict['<PAD>']) + ' ')
        wfile.write('\n')
    rfile.close()
    wfile.close()
    

chn_word_index_dict = train_word2index('train_source_8000.txt')
eng_word_index_dict = train_word2index('train_target_8000.txt')
test_word2index('dev_source_1000.txt', chn_word_index_dict)
test_word2index('dev_target_1000.txt', eng_word_index_dict)
test_word2index('test_source_1000.txt', chn_word_index_dict)
test_word2index('test_target_1000.txt', eng_word_index_dict)