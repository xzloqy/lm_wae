#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-22 上午11:47
# @Author  : Xinxin Zhang
import sys

import os

# 得到当前根目录
parent_path = os.getcwd() # 返回当前工作目录
sys.path.append(parent_path) # 添加自己指定的搜索路径

from base.data.base_data_builder import BaseDataBuilder
from base.data.base_data_loader import Dataloader
from base.config.base_config import BaseConfig
from utils.tools import Voc

import json
import string
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

class DataBuilderCommon(BaseDataBuilder):
    def __init__(self, config, ):
        super(DataBuilderCommon, self).__init__(type='train valid test')
        self.config = config
        self.rate = {'train: valid: test': []}

        self.src_data = {'train' :{'text' :[], 'text2index' :[]},
                         'valid' :{'text' :[], 'text2index' :[]},
                         'test' :{'text' :[], 'text2index' :[]}}
        # self.tgt_data = {'train' :{'text' :[], 'text2index' :[]},
        #                  'valid' :{'text' :[], 'text2index' :[]},
        #                  'test' :{'text' :[], 'text2index' :[]}}
        self.src_vocab = {'vocab' :[], 'vocab2index' :{}}
        # self.tgt_vocab = {'vocab' :[], 'vocab2index' :{}}

    def initAttr(self):
        if os.path.isfile(self.config.saved_data):
            self.loadData()
        else:
            self.readRawData()
            self.buildVocabulary()
            self.convertTextToIndex()
            # self.getMiniData()
            # self.sample()
            self.buildDataloaderFormatData()
            self.saveData()

    def getMaxLength(self):
        src_len = []
        # tgt_len = []
        for type in self.types:
            src_list = []
            # tgt_list = []
            for lst in self.dataloader_format[type]:
                src_list.append(len(lst['lan1_src']))
                # tgt_list.append(len(lst['lan2_tgt']))
            print("the max length of {} data is :{}-{}".format(type, self.config.src, max(src_list)))
            src_len += src_list
            # tgt_len += tgt_list

        counter_src = Counter()
        # counter_tgt = Counter()
        counter_src.update(src_len)
        # counter_tgt.update(tgt_len)
        # freq_en = sorted(counter_src.items(), key=lambda tup: tup[0])
        # freq_pt = sorted(counter_tgt.items(), key=lambda tup: tup[0])
        x_src = counter_src.keys()
        y_src = counter_src.values()
        # x_tgt = counter_tgt.keys()
        # y_tgt = counter_tgt.values()

        plt.ion()
        plt.figure('the contribution of length')

        plt.subplot(211)
        plt.title('the contribution of src_len')
        plt.scatter(x_src, y_src, label='src', color='r', marker='o', )
        plt.grid(True)

        # plt.subplot(212)
        # plt.title('the contribution of tgt_len')
        # plt.scatter(x_tgt, y_tgt, label='tgt', color='b', marker='*', )
        # plt.grid(True)

        plt.tight_layout()
        # plt.subplots_adjust(hspace=0.5)
        plt.ioff()
        plt.show()

    def readRawData(self):
        def read_corpus(file_path, process_punct=False):
            data = []
            trantab = str.maketrans({key: ' ' + key for key in string.punctuation})
            for line in open(file_path):
                if process_punct:
                    # sent = line.translate(trantab).strip().lower().split(' ')[:self.config.max_seq_len]
                    sent = line.translate(trantab).strip().lower().split(' ')
                else:
                    sent = line.strip().lower().split(' ')
                sent = ['<s>'] + sent + ['</s>']
                data.append(sent)
            return data
        print('~' * 15, ' begin to read raw data ', '~' * 15)
        for type in self.types:
            self.src_data[type]['text'] = read_corpus(self.config.src_data_path.format(type), self.config.process_punct)
            # self.tgt_data[type]['text'] = read_corpus(self.config.tgt_data_path.format(type), self.config.process_punct)
            # assert len(self.src_data[type]['text']) == len(self.tgt_data[type]['text']), \
            #     'Note:the length of src must be equal to tgt!'
        print('~' * 15, ' finished read raw data ' , '~' * 15)

    def buildVocabulary(self):
        def read_voc(path):
            vocab_name, vocab2index_name = [], {}
            with open(path, 'r', encoding='utf-8') as f:
                for vocab in f.readlines():
                    vocab_name.append(vocab.replace('\n', ''))
                    vocab2index_name[vocab.replace('\n', '')] = len(vocab_name) - 1
            print('=' * 15, '词典读取完成，共 {} 个词'.format(len(vocab_name)), '=' * 15)
            return vocab_name, vocab2index_name

        def build_voc(path, text_name):
            voc = Voc(text_name, self.config.min_vocab_freq)
            vocabs, vocab2index = voc.getVocabulary()
            with open(path, 'w', encoding='utf-8') as f:
                for vocab in vocabs:
                    f.write(vocab + '\n')
            print('=' * 15, '词典创建完成，共 {} 个词'.format(len(vocabs)), '=' * 15)
            return vocabs, vocab2index

        src_vocab_path = self.config.src_vocab_path
        # tgt_vocab_path = self.config.tgt_vocab_path
        if os.path.isfile(src_vocab_path):
            self.src_vocab['vocab'], self.src_vocab['vocab2index'] = read_voc(src_vocab_path)
            # self.tgt_vocab['vocab'], self.tgt_vocab['vocab2index'] = read_voc(tgt_vocab_path)
        else:
            src_data = self.src_data['train']['text'] + self.src_data['valid']['text'] + self.src_data['test']['text']
            # tgt_data = self.tgt_data['train']['text'] + self.tgt_data['valid']['text'] + self.tgt_data['test']['text']
            self.src_vocab['vocab'], self.src_vocab['vocab2index'] = build_voc(src_vocab_path, src_data)
            # self.tgt_vocab['vocab'], self.tgt_vocab['vocab2index'] = build_voc(tgt_vocab_path, tgt_data)
            self.src_vocab['vocab'], self.src_vocab['vocab2index'] = read_voc(src_vocab_path)
            # self.tgt_vocab['vocab'], self.tgt_vocab['vocab2index'] = read_voc(tgt_vocab_path)

        print('~' * 15, ' finished building vocabulary, src_size = %d' %
        (len(self.src_vocab['vocab'])), '~' * 15)

    def convertTextToIndex(self):
        def getIndex(lst, vocab2index):
            word2index = []
            for sentence in lst:
                index = []
                for word in sentence:
                    index.append(vocab2index.get(word) if word in vocab2index.keys() else vocab2index.get('<unk>'))
                word2index.append(index)
            return word2index

        for t in self.types:
            self.src_data[t]['text2index'] = getIndex(self.src_data[t]['text'], self.src_vocab['vocab2index'])
            # self.tgt_data[t]['text2index'] = getIndex(self.tgt_data[t]['text'], self.tgt_vocab['vocab2index'])

        print('~' * 15, ' finished converting src text to index ', '~' * 15)

    def saveData(self):
        save_path = self.config.saved_data
        data_dict = {
            # 'src_data': self.src_data,
            # 'tgt_data': self.tgt_data,
            'src_vocab': self.src_vocab,
            # 'tgt_vocab': self.tgt_vocab,
            'dataloader_format': self.dataloader_format
        }
        json_str = json.dumps(data_dict)
        with open(save_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json_str)

        print('=' * 15, 'finished save data at {}'.format(save_path), '=' * 15)

    def loadData(self):
        load_path = self.config.saved_data
        with open(load_path, 'r', encoding='utf-8') as json_file:
            data_dict = json.load(json_file)
        # self.src_data = data_dict['src_data']
        # self.tgt_data = data_dict['tgt_data']
        self.src_vocab = data_dict['src_vocab']
        # self.tgt_vocab = data_dict['tgt_vocab']
        self.dataloader_format = data_dict['dataloader_format']
        print('=' * 10, 'finished load data from {}'.format(load_path), '=' * 15)

    def buildDataloaderFormatData(self):
        print('~' * 15, ' begin to build dataloader format data ')
        for t in self.types:
            for i in range(len(self.src_data[t]['text2index'])):
                d = {}
                d['lan1_src'] = self.src_data[t]['text2index'][i][:-1]
                d['lan1_tgt'] = self.src_data[t]['text2index'][i][1:]
                # d['lan2_src'] = self.tgt_data[t]['text2index'][i][:-1]
                # d['lan2_tgt'] = self.tgt_data[t]['text2index'][i][1:]
                self.dataloader_format[t].append(d)
        if self.config.sorted:
            for t in self.types:
                self.dataloader_format[t] = sorted(self.dataloader_format[t], key=lambda i: len(i['src']), reverse=True)
            print('sorted')
        print('~' * 15, ' finished building dataloader format data ')

    def sample(self):
        print('~' * 15, ' begin to random sampling ', '~' * 15)
        x_a, self.src_data['train']['text2index'] = train_test_split(self.src_data['train']['text2index'],  test_size=0.03125, random_state=1)
        print('~' * 15, ' finished random sampling, src num:{} '.format(len(self.src_data['train']['text2index'])), '~' * 15)


if __name__ == '__main__':
    config_param = [
                    ('data', 'min_vocab_freq', 'int'),
                    ('data', 'max_length', 'int'),
                    ('data', 'sorted', 'bool'),
                    ('data', 'process_punct', 'bool'),
                    # ('data', 'tgt_data_path', 'str'),
                    ('data', 'src_data_path', 'str'),
                    # ('data', 'tgt_vocab_path', 'str'),
                    ('data', 'src_vocab_path', 'str'),
                    ('data', 'saved_data', 'str'),
                    ('language', 'src', 'str'),
                    # ('language', 'tgt', 'str'),
                    ]
    conf_path = 'machine_translation/config/config.conf'

    data_conf = BaseConfig(conf_path, config_param)
    data_conf.src = data_conf.src.replace(data_conf.base_dir, '')
    # data_conf.tgt = data_conf.tgt.replace(data_conf.base_dir, '')
    data_conf.src_vocab_path = data_conf.src_vocab_path.format(data_conf.min_vocab_freq)
    # data_conf.tgt_vocab_path = data_conf.tgt_vocab_path.format(data_conf.min_vocab_freq)
    data_conf.saved_data = data_conf.saved_data.format(data_conf.src, data_conf.max_length)
    # information = data_conf.__str__()
    # print(information)
    # print('\n')
    data = DataBuilderCommon(data_conf)
    data.initAttr()
    # data.readRawData()
    # data.buildVocabulary()
    # data.splitDataset()
    # data.convertTextToIndex()
    # train_dataloader = Dataloader(data.dataloader_format['train'])
    # train_iter = train_dataloader.createBatches(4,0)
    # valid_dataloader = Dataloader(data.dataloader_format['valid'])
    # valid_iter = valid_dataloader.createBatches(4,0)

    # data.sample()

    data.getMaxLength()

    print(1)



