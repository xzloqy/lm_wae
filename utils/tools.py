#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-9 下午3:18
# @Author  : Xinxin Zhang
import torch
from collections import Counter

class Voc(object):
    def __init__(self, originalText_list, min_vocab_freq=1):
        self.originalText_list = originalText_list
        self.min_vocab_freq = min_vocab_freq

        self.specials = ['<pad>', '<unk>', '<s>', '</s>']
        self.vocab = []
        self.word2index = {}

        self.num_words = 2

    def getVocabulary(self,):
        def trim(word2count, min_count):
            keep_words = []
            c = 0
            for k, v in word2count:
                if v >= min_count:
                    keep_words.append(word2count[c])
                    c += 1
            return keep_words
        counter = Counter()
        # for text in self.originalText_list:
        #     for sentence in text:
        #         counter.update(sentence)
        for sentence in self.originalText_list:
            counter.update(sentence[1:-1])
        word2count = sorted(counter.items(), key=lambda tup: tup[0])
        word2count.sort(key=lambda tup: tup[1], reverse=True)
        word2count = trim(word2count, self.min_vocab_freq)
        self.vocab = [voc for voc, _ in word2count]
        for i in range(len(self.specials)):
            self.vocab.insert(i,self.specials[i])
        for idx in range(len(self.vocab)):
            self.word2index[self.vocab[idx]] = idx

        return self.vocab, self.word2index

def logSumExp(vec):
    '''
    This function calculates the score explained above for the forward algorithm
    vec 2D: 1 * tagset_size
    '''
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def argmax(vec):
    '''
    This function returns the max index in a vector
    '''
    _, idx = torch.max(vec, 1)
    return toScalar(idx)

def toScalar(var):
    '''
    Function to convert pytorch tensor to a scalar
    '''
    return var.view(-1).data.tolist()[0]

def getTextLen(sent_lstm_length):
    text_length = sum(sent_lstm_length)
    for i in range(text_length.size(0)):
        a = []
        for lst in sent_lstm_length:
            if len(lst) > 1:
                a.append(max(lst))
            else:
                a.append(lst[0])
        text_length[i] = sum(a)
    return text_length

def getTarget(inputs):
    batch_size, sent_num, word_num = inputs[0].size()
    target = []
    for i in range(sent_num):
        temp_inputs = inputs[0][:, i:i+1, :]
        temp_len = inputs[1][:, i:i+1]
        sent_inputs = temp_inputs.contiguous().view(-1, word_num), temp_len.contiguous().view(-1)

        tgt, lengths = sent_inputs
        batch_size = tgt.size(0)
        temps = []
        for i in range(batch_size):
            temp = tgt[i][:max(lengths)]
            temps.append(temp)
        tgt = torch.stack(temps)
        target.append(tgt)
    targets = torch.cat(target,1)
    return targets

def getMaxProbResult(input, ix_to_tag):
    index = 0
    for i in range(1, len(input)):
        if input[i] > input[index]:
            index = i
    return ix_to_tag[index]

def convert_text2idx(examples, word2idx):
    return [[word2idx[w] if w in word2idx else 1     # <unk> : 1
            for w in sent] for sent in examples]

def convert_idx2text(example, idx2word):
    words = []
    for i in example:
        if i.item() == 3:              # <eos> : 3
            break
        words.append(idx2word[i.item()])
    return ' '.join(words)

def getSrc(lst, vocab_list):
    sent = []
    for idx in lst:
        sent.append(vocab_list.index(idx))
    return sent

def convertTocken2Word(batch_size, pred, true, tgt_vocab, hypotheses, references):
    for i in range(batch_size):
        pred_idxs = [idx for idx in pred[i] if
                     idx not in (
                         tgt_vocab.index('<pad>'), tgt_vocab.index('<s>'), tgt_vocab.index('</s>'))]
        pred_line = [tgt_vocab[idx] for idx in pred_idxs]

        true_idxs = [idx for idx in true[i] if
                     idx not in (
                         tgt_vocab.index('<pad>'), tgt_vocab.index('<s>'), tgt_vocab.index('</s>'))]
        true_line = [tgt_vocab[idx] for idx in true_idxs]

        hypotheses.append(pred_line)
        references.append(true_line)

        # if step % self.config.print_freq == 0 and i % 2 == 0:
        #     print('true:{}'.format(' '.join(true_line)))
        #     print('pred:{} \n'.format(' '.join(pred_line)))

def input_transpose(sents, pad_token, max_len):
    batch_size = len(sents)
    sents_t = []
    masks = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in range(batch_size)])

    return sents_t, masks

def get_n_correct(pred, true):
    pred = pred.max(2)[1].view(-1)
    labels = true.contiguous().view(-1)
    non_pad_mask = labels.ne(0)
    n_correct = pred.eq(labels).masked_select(non_pad_mask).sum().item()
    return n_correct