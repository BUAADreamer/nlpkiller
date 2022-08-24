import torch

from nlpkiller import *
from torch import nn

import nlpkiller as nlpk

is_train = False
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, nlpk.try_gpu()
train_iter, src_vocab, tgt_vocab = nlpk.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
net = nlpk.EncoderDecoder(encoder, decoder)
if is_train:
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
net.load_state_dict(torch.load("models/rnns2s.pth"))
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
    