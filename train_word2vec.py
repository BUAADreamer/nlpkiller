from torch import nn
import math
import torch

import nlpkiller as nlpk


def test_word2vec(is_train=False):
    def get_similar_tokens(query_token, k, embed):
        W = embed.weight.data
        x = W[vocab[query_token]]
        # 计算余弦相似性。增加1e-9以获得数值稳定性
        cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                          torch.sum(x * x) + 1e-9)
        topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype('int32')
        for i in topk[1:]:  # 删除输⼊词
            print(f'cosine sim={float(cos[i]):.3f}: {vocab.idxs2tokens(i)}')

    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = nlpk.load_data_ptb(batch_size, max_window_size,
                                          num_noise_words)
    embed_size = 100
    net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(vocab),
                                     embedding_dim=embed_size))
    lr, num_epochs = 0.002, 5
    if is_train:
        nlpk.train_word2vec(net, data_iter, lr, num_epochs)
        torch.save(net.state_dict(), "models/word2vec.pth")
    else:
        net.load_state_dict(torch.load("models/word2vec.pth"))
    get_similar_tokens('chip', 3, net[0])


# test_word2vec(True)
# test_word2vec()
glove_6b50d = nlpk.TokenEmbedding('glove.6b.50d')
print(len(glove_6b50d))
print(glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367])
print(nlpk.get_similar_tokens('baby', 3, glove_6b50d))
print(nlpk.get_analogy('man', 'woman', 'son', glove_6b50d))
print(nlpk.get_analogy('beijing', 'china', 'tokyo', glove_6b50d))
print(nlpk.get_analogy('bad', 'worst', 'big', glove_6b50d))
print(nlpk.get_analogy('do', 'did', 'go', glove_6b50d))
