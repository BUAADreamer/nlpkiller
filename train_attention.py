import torch
from torch import nn
import nlpkiller as nlpk

##############
# 构造数据集
##############
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)  # 排序后的训练样本
f = lambda x: 2 * torch.sin(x) + x ** 0.8
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数


def Nadaraya_Watson():
    '''
    非参数模型
    '''
    # X_repeat的形状:(n_test,n_train),
    # 每⼀⾏都包含着相同的测试输⼊（例如：同样的查询）
    # X_repeat每一行都是同一个查询值 [[0,0,...,0],[0.1,0.1,...,0.1],...]
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # print(X_repeat)
    # x_train包含着键。attention_weights的形状：(n_test,n_train),
    # 每⼀⾏都包含着要在给定的每个查询的值（y_train）之间分配的注意⼒权重
    attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
    # print(attention_weights.shape, y_train.shape)
    # y_hat的每个元素都是值的加权平均值，其中的权重是注意⼒权重
    y_hat = torch.matmul(attention_weights, y_train)
    # print(attention_weights, y_train)
    # print(y_hat.shape)


# Nadaraya_Watson()

######
# [[1,2],[3,4]]*[1,2]
# print(torch.matmul(X, Y))  # [5,11]
######
X = torch.arange(1, 5).reshape(2, 2)
Y = torch.arange(1, 3)


def train_NWKernelRegression():
    '''
    参数模型
    '''
    # X_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输⼊
    X_tile = x_train.repeat((n_train, 1))
    # Y_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输出
    Y_tile = y_train.repeat((n_train, 1))
    # keys的形状:('n_train'，'n_train'-1)
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # values的形状:('n_train'，'n_train'-1)
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    net = nlpk.NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    # keys的形状:(n_test，n_train)，每⼀⾏包含着相同的训练输⼊（例如，相同的键）
    keys = x_train.repeat((n_test, 1))
    # value的形状:(n_test，n_train)
    values = y_train.repeat((n_test, 1))
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()
    print(y_hat)


def test_AdditiveAttention():
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    # values的⼩批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
        2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    attention = nlpk.AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                                       dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))


# test_AdditiveAttention()

def test_Bahdanau(is_train=False):
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, nlpk.try_gpu()
    train_iter, src_vocab, tgt_vocab = nlpk.load_data_nmt(batch_size, num_steps)
    encoder = nlpk.Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = nlpk.Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = nlpk.EncoderDecoder(encoder, decoder)
    if is_train:
        nlpk.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
        torch.save(net.state_dict(), "models/bahdanau.pth")
    else:
        net.load_state_dict(torch.load("models/bahdanau.pth"))
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = nlpk.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ', f'bleu {nlpk.bleu(translation, fra, k=2):.3f}')


# test_Bahdanau(True)
# test_Bahdanau()

def test_transformer(is_train=False):
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, nlpk.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    train_iter, src_vocab, tgt_vocab = nlpk.load_data_nmt(batch_size, num_steps)
    encoder = nlpk.TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = nlpk.TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = nlpk.EncoderDecoder(encoder, decoder)
    if is_train:
        nlpk.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
        torch.save(net.state_dict(), "models/transformer.pth")
    else:
        net.load_state_dict(torch.load("models/transformer.pth"))
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = nlpk.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ', f'bleu {nlpk.bleu(translation, fra, k=2):.3f}')


# test_transformer(True)
test_transformer()
