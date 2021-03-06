from nlpkiller import *
from torch import nn


def test_text2ngram():
    # 1.下载读取字符串数据
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/' + 'timemachine.txt'
    crawler = Crawler(url)
    text = crawler.download_from_url()
    lines = text.split('\n')
    # 2.词元化
    tokens = tokenize(lines, 'word', 1)
    # 3.词表
    # 一元语法
    unigram_voc = Vocabulary(tokens)
    # print(unigram_voc.idx2token[:10])
    print(unigram_voc.sort_freqs[:10])
    # 二元语法
    bigram_tokens = [pair for pair in zip(tokens[:-1], tokens[1:])]
    bigram_voc = Vocabulary(bigram_tokens)
    print(bigram_voc.sort_freqs[:10])
    # 三元语法
    trigram_tokens = [triple for triple in zip(tokens[:-2], tokens[1:-1], tokens[2:])]
    trigram_voc = Vocabulary(trigram_tokens)
    print(trigram_voc.sort_freqs[:10])


def test_train_simple_rnn():
    batch_size, num_steps, max_tokens = 32, 35, 10
    # corpus, vocab = load_corpus_time_machine(max_tokens)
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    # # 256个隐藏单元的单隐藏层RNN
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    # # 初始化隐状态
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    num_epochs, lr = 500, 1
    train_rnn(net, train_iter, vocab, lr, num_epochs, device='cpu')


test_train_simple_rnn()
