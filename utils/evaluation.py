#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-24 下午8:37
# @Author  : Xinxin Zhang
import numpy as np
import torch
import torch.nn.functional as F

from collections import Counter
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics.pairwise import cosine_similarity

def get_bleu(references, hypotheses):
    # compute BLEU
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    if hypotheses[0][0] == '<s>':
        hypotheses = [hyp[1:-1] for hyp in hypotheses]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp for hyp in hypotheses])

    return bleu_score


def get_acc(references, hypotheses, acc_type='word_acc'):
    assert acc_type == 'word_acc' or acc_type == 'sent_acc'
    cum_acc = 0.

    for ref, hyp in zip(references, hypotheses):
        ref = ref[1:-1]
        hyp = hyp[1:-1]
        if acc_type == 'word_acc':
            acc = len([1 for ref_w, hyp_w in zip(ref, hyp) if ref_w == hyp_w]) / float(len(hyp) + 1e-6)
        else:
            acc = 1. if all(ref_w == hyp_w for ref_w, hyp_w in zip(ref, hyp)) else 0.
        cum_acc += acc

    acc = cum_acc / len(hypotheses)
    return acc


def cal_performance(criterion, out, labels):
    # loss = F.cross_entropy(out.view(-1, out.size(-1)), labels,
    #                        ignore_index=0,
    #                        reduction='sum')
    # labels = tgt_batch[:, 1:].permute(1, 0).contiguous().view(-1)
    loss = criterion(out.view(-1, out.size(-1)), labels.view(-1))/out.size(0)
    pred = out.max(2)[1].view(-1)
    labels = labels.contiguous().view(-1)
    non_pad_mask = labels.ne(0)
    n_correct = pred.eq(labels)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct

def get_metrics(hypotheses, references, preds, tgt_batch, tgt_vocab, step, print_freq):
    batch_size = tgt_batch.size(0)
    for i in range(batch_size):
        pred_idxs = [idx for idx in preds[i] if
                     idx not in (tgt_vocab.index('<pad>'), tgt_vocab.index('<s>'),
                                 tgt_vocab.index('</s>'))]
        pred_line = [tgt_vocab[idx] for idx in pred_idxs]
        true_idxs = [idx for idx in tgt_batch[i] if
                     idx not in (tgt_vocab.index('<pad>'), tgt_vocab.index('<s>'),
                                 tgt_vocab.index('</s>'))]
        true_line = [tgt_vocab[idx] for idx in true_idxs]

        hypotheses.append(pred_line)
        references.append(true_line)

        if step % print_freq == 0 and i % (batch_size / 2) == 0:
            print('true:{}'.format(' '.join(true_line)))
            print('pred:{} \n'.format(' '.join(pred_line)))


