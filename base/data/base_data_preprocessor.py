#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-9 下午3:00
# @Author  : Xinxin Zhang
"""
base class
"""
class DataPreprocessor(object):
    def __init__(self, name, raw_data_path, vocab_path, model_data_path, labeled=True, **kw):
        self.name = name
        self.raw_data_path = raw_data_path
        self.vocab_path = vocab_path.format(self.name)+'/{}_vocab.txt'
        self.model_data_path = model_data_path.format(self.name)
        self.labeld = labeled
        for k, w in kw.items():
            setattr(self, k, w)

        self.raw_data = {}
        self.dataloader_format = []

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



