#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-9 下午3:11
# @Author  : Xinxin Zhang
import torch.nn as nn

class BaseModel(nn.Module):
    """
    BaseModel
    """
    def __init__(self, model_config):
        super(BaseModel, self).__init__()
        self.model_config = model_config

    def encoder(self):
        """
        encoder
        """
        raise NotImplementedError

    def decoder(self):
        """
        decoder
        """
        raise NotImplementedError

    def forward(self, *input):
        """
        forward
        """
        raise NotImplementedError

    def __repr__(self):
        main_string = super(BaseModel, self).__repr__()
        num_parameters = sum([p.nelement() for p in self.parameters()])
        main_string += "\nNumber of parameters: {}\n".format(num_parameters)
        return main_string
