#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-9 下午3:29
# @Author  : Xinxin Zhang
"""
创建一个Embedd类，经初始化后可读取vocabulary.txt的词表，并将词表中的单词用bert转化为词向量，保存在vocab_vector.txt
"""
from bert_serving.client import BertClient
read_path = '/home/zutnlp/xiangzheng/WAEforMT/zutnlp_research/xzwae/data/snli/snli_sentences_1000.txt'
save_path = '/home/zutnlp/xiangzheng/WAEforMT/zutnlp_research/xzwae/data/snli/snli_small_vector.txt'
class Embedder(object):
    """
    word_vector read path = 'vocabulary.txt'
    word_vector save path = 'vocab_vector.txt'
    durt_data : which is not in bert's vocab.txt
    """
    def __init__(self, read_path, save_path, durt_data):
        self.read_path = read_path
        self.save_path = save_path
        self.durt_data = durt_data
        self.vocab = []
        self.vocab_temp = []
        self.vocab_vector_dict = {}

    def getEmbedding(self):
        self.getVocab()
        self.buildFraudVocab()
        self.buildWord2vectorDict()
        self.saveWord2vectorDict()

    def getVocab(self):
        with open(self.read_path, 'r', encoding='utf-8') as f:
            for i in f.readlines():
                self.vocab.append(i.replace('\n',''))
        print('finished read vocabulary, vocabulary size is {}. '.format(len(self.vocab)))

    def buildFraudVocab(self):

        for word in self.vocab:
            if word in durt:
                self.vocab_temp.append('<unk>')
            else:
                self.vocab_temp.append(word)
        print('finished build fraud vocabulary. ')

    def buildWord2vectorDict(self):
        bc = BertClient()
        vocab_vextor = bc.encode(self.vocab_temp)
        for i in range(len(vocab_vextor)):
            key = self.vocab_temp[i]
            value = vocab_vextor[i].tolist()
            self.vocab_vector_dict[key] = value
        print('finished build word2vector_dict. ')

    def saveWord2vectorDict(self):
        with open(self.save_path, 'w') as output:
            for word in self.vocab_vector_dict.keys():
                vector = self.vocab_vector_dict.get(word)
                str_ = ''
                for i in vector:
                    str_ += str(i) + ' '
                str_line = str_.strip(' ')
                output.write(word + " " + str_line + '\n')
        print('finished save word to vector')

# ~~~~~~~~~~~~~~~~~~~~~~ Example:how to use ~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    durt = ['', ' ', '\u3000', '\ue236', '\x04', '\t']
    # em = Embedder('word_vector read path','word_vector save path', durt)
    em = Embedder(read_path,save_path, durt)
    em.getEmbedding()

