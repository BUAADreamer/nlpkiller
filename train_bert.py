import torch
from torch import nn

import nlpkiller as nlpk


def test_bert(is_train=False):
    batch_size, max_len = 512, 64
    train_iter, vocab = nlpk.load_data_wiki(batch_size, max_len)
    net = nlpk.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                         ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                         num_layers=2, dropout=0.2, key_size=128, query_size=128,
                         value_size=128, hid_in_features=128, mlm_in_features=128,
                         nsp_in_features=128)
    devices = nlpk.try_all_gpus()
    loss = nn.CrossEntropyLoss()
    if is_train:
        nlpk.train_bert(train_iter, net, loss, len(vocab), devices, 50)
        torch.save(net.state_dict(), "models/bertwikitext2.pth")
    else:
        net.load_state_dict(torch.load("models/bertwikitext2.pth"))
    tokens_a = ['a', 'crane', 'is', 'flying']
    encoded_text = nlpk.get_bert_encoding(net, tokens_a, vocab, devices)
    # 词元：'<cls>','a','crane','is','flying','<sep>'
    encoded_text_cls = encoded_text[:, 0, :]
    encoded_text_crane = encoded_text[:, 2, :]
    print(encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3])
    tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
    encoded_pair = nlpk.get_bert_encoding(net, tokens_a, vocab, devices, tokens_b)
    # 词元：'<cls>','a','crane','driver','came','<sep>','he','just',
    # 'left','<sep>'
    encoded_pair_cls = encoded_pair[:, 0, :]
    encoded_pair_crane = encoded_pair[:, 2, :]
    print(encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3])


test_bert(True)
# test_bert()
