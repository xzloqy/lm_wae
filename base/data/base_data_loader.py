#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-9 下午2:52
# @Author  : Xinxin Zhang
import torch
import torch.utils.data as Data
import math
import random
from torch.autograd import Variable

class Pack(dict):
    def __getattr__(self, name):
        return self.get(name)
    def cuda(self, device=None):
        """
        cuda
        """
        pack = Pack()
        for k, v in self.items():
            if isinstance(v, tuple):
                pack[k] = tuple(x.cuda(device) for x in v)
            else:
                pack[k] = v.cuda(device)
        return pack

def list2tensor(X):
    """
    list2tensor
    """
    size = maxLens(X)
    if len(size) == 1:
        tensor = torch.tensor(X)
        return tensor

    tensor = torch.zeros(size, dtype=torch.long)
    lengths = torch.zeros(size[:-1], dtype=torch.long)
    if len(size) == 2:
        for i, x in enumerate(X):
            l = len(x)
            tensor[i, :l] = torch.tensor(x)
            lengths[i] = l
    else:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                l = len(x)
                tensor[i, j, :l] = torch.tensor(x)
                lengths[i, j] = l

    return tensor, lengths

def maxLens(X):
    """
    max_lens
    """
    if not isinstance(X[0], list):
        return [len(X)]
    elif not isinstance(X[0][0], list):
        return [len(X), max(len(x) for x in X)]
    elif not isinstance(X[0][0][0], list):
        return [len(X), max(len(x) for x in X),
                max(len(x) for xs in X for x in xs)]
    else:
        raise ValueError(
            "Data list whose dim is greater than 3 is not supported!")

class Dataloader(Data.Dataset):
    """
    dataloader
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(device=-1):
        """
        collate_fn
        """
        def collate(data_list):
            """
            collate
            """
            batch = Pack()
            for key in data_list[0].keys():
                batch[key] = list2tensor([x[key] for x in data_list])
            if device >= 0:
                batch = batch.cuda(device=device)
            return batch
        return collate

    def createBatches(self, batch_size=1, shuffle=True, device=0):
        """
        create_batches
        """
        loader = Data.DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False)
        return loader





#
# class Dataloader(object):
#     """Class to Load Language Pairs and Make Batch
#     srcFilename = tgtFilename
#         .txt格式，每行一句文本
#     """
#
#     def __init__(self, srcFilename, tgtFilename, batch_size, cuda=False, volatile=False):
#         # Need to reload every time because memory error in pickle
#         srcFile = open(srcFilename)
#         tgtFile = open(tgtFilename)
#         src = []
#         tgt = []
#         nb_pairs = 0
#         while True:
#             src_line = srcFile.readline()
#             tgt_line = tgtFile.readline()
#             if src_line == '' and tgt_line == '':
#                 break
#             src_ids = list(map(int, src_line.strip().split()))
#             tgt_ids = list(map(int, tgt_line.strip().split()))
#             if 0 in src_ids or 0 in tgt_ids:
#                 continue
#             if len(src_ids) > 0 and len(src_ids) <= 64 and len(tgt_ids) > 0 and len(tgt_ids) <= 64:
#                 src.append(src_ids)
#                 tgt.append(tgt_ids)
#                 nb_pairs += 1
#         print('%d pairs are converted in the data' % nb_pairs)
#         srcFile.close()
#         tgtFile.close()
#         sorted_idx = sorted(range(nb_pairs), key=lambda i: len(src[i]))
#         self.src = [src[i] for i in sorted_idx]
#         self.tgt = [tgt[i] for i in sorted_idx]
#         self.batch_size = batch_size
#         self.nb_pairs = nb_pairs
#         self.nb_batches = math.ceil(nb_pairs / batch_size)
#         self.cuda = cuda
#         self.volatile = volatile
#
#     def __len__(self):
#         return self.nb_batches
#
#     def _shuffle_index(self, n, m):
#         """Yield indexes for shuffling a length n seq within every m elements"""
#         indexes = []
#         for i in range(n):
#             indexes.append(i)
#             if (i + 1) % m == 0 or i == n - 1:
#                 random.shuffle(indexes)
#                 for index in indexes:
#                     yield index
#                 indexes = []
#
#     def shuffle(self, m):
#         """Shuffle the language pairs within every m elements
#
#         This will make sure pairs in the same batch still have similar length.
#         """
#         shuffled_indexes = self._shuffle_index(self.nb_pairs, m)
#         src, tgt = [], []
#         for index in shuffled_indexes:
#             src.append(self.src[index])
#             tgt.append(self.tgt[index])
#         self.src = src
#         self.tgt = tgt
#
#     def _wrap(self, sentences):
#         """Pad sentences to same length and wrap into Variable"""
#         max_size = max([len(s) for s in sentences])
#         out = [s + [0] * (max_size - len(s)) for s in sentences]
#         out = torch.LongTensor(out)
#         if self.cuda:
#             out = out.cuda()
#         return Variable(out, volatile=self.volatile)
#
#     def __getitem__(self, i):
#         """Generate the i-th batch and wrap in Variable"""
#         src_batch = self.src[i * self.batch_size:(i + 1) * self.batch_size]
#         tgt_batch = self.tgt[i * self.batch_size:(i + 1) * self.batch_size]
#         return self._wrap(src_batch), self._wrap(tgt_batch)

if __name__ == '__main__':
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    print("Building Dataloader ...")

    train_path = '/home/zutnlp/zhangxinxin/low_resourceNMT/Transformer-py-master/translation-data/newstest2016.'
    traindataloader = Dataloader(train_path + "en.id", train_path + "de.id", 4, cuda=True)