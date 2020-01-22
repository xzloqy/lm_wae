#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-21 下午5:52
# @Author  : Xinxin Zhang
"""
base class
"""
class BaseDataBuilder(object):
    # def __init__(self, raw_data_path, vocab_path, model_data_path, labeled=True, **kw):
    def __init__(self, type='train valid test', **kw):
        # self.name = name
        self.base_dir = '/home/zutnlp/zhangxinxin/MachineTranslation/MyCode/zutnlp_research/zutnlp_research/'
        # self.raw_data_path = self.base_dir + raw_data_path
        # self.vocab_path = self.base_dir + vocab_path
        # self.model_data_path = self.base_dir + model_data_path

        # self.labeld = labeled
        for k, w in kw.items():
            setattr(self, k, w)

        assert type == 'train valid test' or type == 'train valid ' or type == 'test', \
            'please choose type from :["train valid test", "train valid", "test"]'
        self.types = type.strip().split()

        self.raw_data = {}
        self.dataloader_format = {}
        if len(self.types) > 1:
            self.raw_data['all'] = {}
        for t in self.types:
            self.raw_data[t] = {}
            self.dataloader_format[t] = []
        # self.raw_data = {'all':{}, 'train':{}, 'valid':{}, 'test':{}}
        # self.dataloader_format = {'train':[], 'valid':[], 'test':[]}

    def initAttr(self):
        """
        init_attr
        """
        raise NotImplementedError

    def readRawData(self):
        """
        read_raw_data
        """
        raise NotImplementedError

    def buildVocabulary(self):
        """
        build_vocabulary
        """
        raise NotImplementedError

    def convertTextToIndex(self):
        """
        convert_text_to_index
        """
        raise NotImplementedError

    def buildDataloaderFormatData(self):
        """
        build_dataloader_format_data
        """
        raise NotImplementedError

    def saveData(self):
        """
        save_data
        """
        raise NotImplementedError

    def loadData(self):
        """
        load_data
        """
        raise NotImplementedError

