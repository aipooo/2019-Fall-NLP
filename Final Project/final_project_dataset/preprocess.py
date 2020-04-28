

total_size=10000

train_size = 8000
test_size = 1000
dev_size = 1000

source_raw = open('news-commentary-v13.zh-en.zh', 'r', encoding='utf-8')
target_raw = open('news-commentary-v13.zh-en.en', 'r', encoding='utf-8')

source = source_raw.readlines()
target = target_raw.readlines()

source_train = source[:train_size]
target_train = target[:train_size]

source_dev = source[train_size:train_size + dev_size]
target_dev = target[train_size:train_size + dev_size]

source_test = source[train_size+dev_size:train_size+dev_size+test_size]
target_test = target[train_size+dev_size:train_size+dev_size+test_size]

source_out_train = open('dataset_{}/train_source_{}.txt'.format(total_size, train_size), 'w', encoding='utf-8')
target_out_train = open('dataset_{}/train_target_{}.txt'.format(total_size, train_size), 'w', encoding='utf-8')

source_out_dev = open('dataset_{}/dev_source_{}.txt'.format(total_size, dev_size), 'w', encoding='utf-8')
target_out_dev = open('dataset_{}/dev_target_{}.txt'.format(total_size, dev_size), 'w', encoding='utf-8')

source_out_test = open('dataset_{}/test_source_{}.txt'.format(total_size, test_size), 'w', encoding='utf-8')
target_out_test = open('dataset_{}/test_target_{}.txt'.format(total_size, test_size), 'w', encoding='utf-8')

source_out_train.writelines(source_train)
target_out_train.writelines(target_train)

source_out_dev.writelines(source_dev)
target_out_dev.writelines(target_dev)

source_out_test.writelines(source_test)
target_out_test.writelines(target_test)

source_out_train.close()
target_out_train.close()

source_out_dev.close()
target_out_dev.close()

source_out_test.close()
target_out_test.close()

source_raw.close()
target_raw.close()