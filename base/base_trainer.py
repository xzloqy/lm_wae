#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-9 下午3:11
# @Author  : Xinxin Zhang
import os
import torch
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

class BaseTrainer(object):
    def __init__(self, model, optimizer, criterion,
                       train_iter, valid_iter, test_iter,
                       logger, config, **kw):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.test_iter = test_iter

        self.logger = logger
        self.config = config

        self.train_start_message = '=' * 80 + '\nBegin training...'
        self.valid_start_message = "-" * 80 + "\nEvaluating on the valid data"
        self.test_start_message = "*" * 80 + "\nEvaluating on test data"

        self.writer = SummaryWriter()       #  获取一个writer
        # self.SummaryWriter_dir = self.config.SummaryWriter_dir

        for k, w in kw.items():
            setattr(self, k, w)

        self.early_stop_count = 0

    def train(self, epoch):
        """
        train_epoch
        """
        raise NotImplementedError

    def run(self):
        """
        train
        """
        raise NotImplementedError

    def test(self):
        """
        test
        """
        raise NotImplementedError

    def evaluate(self, data_iter):
        """
        evaluate
        """
        raise NotImplementedError

    def saveResults(self):
        """
        save_results
        """
        raise NotImplementedError

    def save(self):
        """
        save
        """
        with open(self.config.best_param_path, 'wb') as f:
            torch.save(self.model, f)
        self.logger.info("~" * 80 + "\n Saved model state to '{}' \n".format(self.config.best_param_path) + "~" * 80 + '\n')

    def load(self):
        """
        load
        """
        if os.path.isfile(self.config.best_param_path):
            self.model = torch.load(self.config.best_param_path)
            self.logger.info(" Loaded model state from '{}' ".format(self.config.best_param_path))
        else:
            self.logger.info(" Invalid model state file: '{}' ".format(self.config.best_param_path))

