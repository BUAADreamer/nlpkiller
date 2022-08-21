import collections

import torch


def preprocess_nmt(text):
    # 不间断空格转换为空格 大写转小写
    # 每个标点/字符之间都有一个空格
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + c if i > 0 and no_space(c, text[i - 1])
           else c for i, c in enumerate(text)]
    return ''.join(out)


def truncate_pad(line, num_steps, padding_token):
    """截断或填充⽂本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def tokenize(lines, token_type='word', dim=2):
    if token_type == 'word':
        res = [line.split() for line in lines]
        if dim == 2:
            return res
        elif dim == 1:
            new_res = []
            for words in res:
                for word in words:
                    new_res.append(word)
            return new_res
    elif token_type == 'char':
        if dim == 2:
            return [list(line) for line in lines]
        elif dim == 1:
            return [c for line in lines for c in list(line)]
    else:
        return []


class Vocabulary:
    """
    文本词表
    """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        # 统计词频得到词频字典
        if reserved_tokens is None: reserved_tokens = []
        if tokens is None: tokens = []
        freqs = self.count_freq(tokens)
        # 词频字典进行二元组排序，即[(s,cnt[s])]进行排序
        self.sort_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        # 抛弃词或者保留字词先初始化id
        self.idx2token = ['<unk>'] + reserved_tokens
        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}
        self.tokens = tokens
        # 对不是抛弃词的词语建立到数字索引的建立
        for token, freq in self.sort_freqs:
            if freq < min_freq: break
            if token not in self.token2idx:
                self.idx2token.append(token)
                self.token2idx[token] = len(self.idx2token) - 1

    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token2idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def idxs2tokens(self, indexs):
        if not isinstance(indexs, (list, tuple)):
            return self.idx2token[indexs]
        return [self.idx2token[index] for index in indexs]

    def count_freq(self, tokens):  # @save
        """统计词元的频率"""
        # 这⾥的tokens是1D列表或2D列表
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # 将词元列表展平成⼀个列表
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    @property
    def unk(self):
        return 0

    @property
    def freqs(self):
        return self.freqs
