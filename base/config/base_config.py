#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-21 下午4:35
# @Author  : Xinxin Zhang
import configparser
import os
class BaseConfig(object):
    def __init__(self, conf_path, congif_params):
        """
        :param conf_path: str
        :param congif_params:[('section', 'key', 'type'),
                              ...,
                              ('section', 'key', 'type')]
        """
        self.project_name = 'zutnlp_research'
        self.base_dir = os.path.abspath(os.path.join(os.getcwd(),'../..'))+'/'
        self.conf_path = conf_path

        self.src_vocab_size = None
        self.src_vocab = None

        # self.tgt_vocab_size = None
        # self.tgt_vocab = None

        for congif_param in congif_params:
            section, key, type = congif_param
            setattr(self, key, self.getConfig(section, key, type))

    def getConfig(self, section, key, type):
        config = configparser.ConfigParser()
        conf_path = self.base_dir + self.conf_path
        try:
            config.read(conf_path)
            if type == 'str':
                return self.base_dir + config.get(section, key)
            elif type == 'int':
                return config.getint(section, key)
            elif type == 'float':
                return config.getfloat(section, key)
            elif type == 'bool':
                return config.getboolean(section, key)
        except Exception as e:
            print('error:%s, please choose the type from ["str", "int", "float", "bool"]' % e)

    def __str__(self):
        return ' \n'.join("%s: %s" % item for item in vars(self).items())



if __name__ == '__main__':
    conf_path = 'base/config/train_config.conf'
    config_param = [('data', 'en_data_path', 'str'),
                    ('data', 'pt_data_path', 'str'),
                    ('data', 'min_vocab_freq', 'int'),
                    ('data', 'max_sent_len', 'int'),
                    ('data', 'en_vocab_path', 'str'),
                    ('data', 'pt_vocab_path', 'str'),
                    ('data', 'saved_data', 'str'),
                    ('mode', 'languageA', 'str'),
                    ('mode', 'languageB', 'str'),
                    ('network params', 'uniform_init', 'float'),
                    ('network params', 'max_seq_len', 'int'),
                    ('network params', 'num_layers', 'int'),
                    ('network params', 'num_heads', 'int'),
                    ('network params', 'ffn_dim', 'int'),
                    ('network params', 'model_dim', 'int'),
                    ('network params', 'dropout', 'float'),
                    ('network params', 'fuse_method', 'str'),
                    ('training params', 'lr', 'float'),
                    ('training params', 'device', 'int'),
                    ('training params', 'epoch', 'int'),
                    ('training params', 'batch_size', 'int'),
                    ('training params', 'max_grad_norm', 'float'),
                    ('training params', 'warmup_steps', 'int'),
                    ('training params', 'valid_freq', 'int'),
                    ('training params', 'display_freq', 'int'),
                    ('training params', 'print_freq', 'int'),
                    ('training params', 'use_RN', 'bool'),
                    ('training params', 'log_dir', 'str'),
                    ('training params', 'best_param_path', 'str'),
                    ('training params', 'test_result_path', 'str'),
                    ('predict params', 'decode_output', 'str'),
                    ('predict params', 'beam_size', 'int'),
                    ('predict params', 'n_best', 'int'),
                    ('predict params', 'max_decode_step', 'int'),
                    ]
    train_conf = BaseConfig(conf_path, config_param)
    information = train_conf.__str__()
    print(information)


