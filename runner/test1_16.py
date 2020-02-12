#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-8-23 下午4:54
# @Author  : Xinxin Zhang
import sys
sys.path.append('../../')
sys.path.append('../')
import random
import math
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import os
from tensorboardX import SummaryWriter
from base.config.base_config import BaseConfig
from base.data.base_data_loader import Dataloader
from base.base_trainer import BaseTrainer
from data.data_builder import DataBuilderCommon
from model.vae.vae import AutoEncoder
from utils.evaluation import cal_performance, get_acc, get_bleu, get_metrics
from utils.tools import get_n_correct
from utils.loss import (recon_loss, total_kld, flow_kld, compute_mmd, 
                    mutual_info, mutual_info_flow,
                    compute_nll)
from data.data import Corpus, get_iterator, PAD_TOKEN, SOS_TOKEN
import warnings

ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
def kl_anneal_function(step, k, x0):

    return float(1/(1+np.exp(-k*(step-x0))))

def weight_schedule(t, start=6000, end=40000):
    return max(min((t - 6000) / 40000, 1), 0)

def idx_to_word(idx, idx2word, pad_idx):
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_idx in sent:
            if word_idx == pad_idx:
                break
            sent_str[i] += idx2word[word_idx] + " "
        sent_str[i] = sent_str[i].strip()
    
    return sent_str

torch.manual_seed(1233)
random.seed(1233)
warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
# use_cuda = False
###############################################################################
# Load Config
###############################################################################
conf_path = 'lm_wae/config/config.conf'
config_param = [
                ('data', 'min_vocab_freq', 'int'),
                ('data', 'max_length', 'int'),
                ('data', 'sorted', 'bool'),
                ('data', 'process_punct', 'bool'),
                ('data', 'input_data_path', 'str'),
                ('data', 'input_vocab_path', 'str'),
                ('data', 'saved_data', 'str'),

                ('dataset', 'dataset', 'str'),
                ('dataset', 'datadir', 'str'),

                ('network params','uniform_init','float'),
                # ('network params','rnn_type','str'),
                ('network params','num_layers','int'),
                ('network params','embed_size','int'),
                ('network params','hidden_size','int'),
                ('network params','latent_size','int'),
                ('network params','word_dropout','float'),
                ('network params','embedding_dropout','float'),
                ('network params','batch_norm','bool'),

                ('training params','epoch','int'),
                ('training params','epoch_size','int'),
                ('training params','lr','float'),
                ('training params','lr_decay','float'),
                ('training params','device','int'),
                ('training params','batch_size','int'),
                ('training params','max_sequence_length','int'),
                ('training params','display_freq','int'),
                ('training params','print_freq','int'),

                ('training params','log_dir','str'),
                ('training params','best_param_path','str'),
                ('training params','test_result_path','str'),
                ('predict params','decode_output','str'),
                ('predict params','beam_size','int'),
                ('predict params','n_best','int'),
                ('predict params','max_decode_step','int'),
                ]
def initConfig(path, param):
    config = BaseConfig(path, param)
    config.dataset = config.dataset.rstrip('/').split('/')[-1]
    config.input_vocab_path = config.input_vocab_path.format(config.min_vocab_freq)
    config.saved_data = config.saved_data.format(config.dataset, config.max_length)
    assert 0 <= config.word_dropout <= 1

    config.log_dir = config.log_dir.format(config.dataset,  config.batch_size, 'without') + '.log'
    config.best_param_path = config.best_param_path.format(config.dataset, config.batch_size, 'without', '') + '.pt'
    config.test_result_path = config.test_result_path.format(config.dataset, config.batch_size, 'without', '') + '.txt'
    # config.decode_output = config.decode_output.format(config.dataset, config.batch_size, 'without', ) + '.txt'
    return config
config = initConfig(conf_path, config_param)
###############################################################################
# Logger definition
###############################################################################
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
fh = logging.FileHandler(config.log_dir)
logger.addHandler(fh)
logger.info(config.__str__())
print('=' * 100)


###############################################################################
# Load Data
###############################################################################
print('Loading corpus...')
# datadir = '~/xiangzheng/machine_translation/data/ptb'
corpus = Corpus(
    config.datadir, max_vocab_size=20000,
    max_length=config.max_length
)
pad_id = corpus.word2idx[PAD_TOKEN]
sos_id = corpus.word2idx[SOS_TOKEN]
config.input_vocab_size = len(corpus.idx2word)
config.rnn_type = 'lstm' # 后做修改加入config.conf
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter = get_iterator(corpus.train, config.batch_size, True,  device)
valid_iter = get_iterator(corpus.valid, config.batch_size, False, device)
test_iter  = get_iterator(corpus.test,  config.batch_size, False, device)

#############################################################################
# Prepare Model
###############################################################################
print('Preparing models...')
model = AutoEncoder(config)
if use_cuda:
    print('Using GPU..')
    model.cuda(device=config.device)
if config.uniform_init:
    print('uniformly initialize parameters [-%f, +%f]' % (config.uniform_init, config.uniform_init))
    for p in model.parameters():
        p.data.uniform_(-config.uniform_init, config.uniform_init)
optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

###############################################################################
# Trainer
###############################################################################
config.tensorboard_logging = True
config.logdir = 'runs'
if config.tensorboard_logging:
    writer = SummaryWriter(os.path.join(config.logdir, ts))
    writer.add_text("model", str(model))
    writer.add_text("config", str(config))
    writer.add_text("ts", ts)
class TrainerEpoch(BaseTrainer):
    def __init__(self, model=None, optimizer=None, criterion=None,
                 train_iter=None, valid_iter=None, test_iter=None,
                 logger=None, config=None, patience=30):
        super(TrainerEpoch, self).__init__(model, optimizer, criterion,
                                      train_iter, valid_iter, test_iter,
                                      logger, config)
        self.patience = patience
        self.epoch = 0
        self.best_epoch = 0
        self.best_iteration = 0
        self.early_stop_count = 0

        self.eval_src_acc = 0

        self.eval_bleu = None
        self.test_bleu = None

        self.best_eval_bleu = 0
        self.best_eval_src_acc = 10000
        self.best_eval_loss = 1000
        self.myloss = {}

    def init_loss(self):
        self.myloss['re_loss'] = 0
        self.myloss['kl_divergence'] = 0
        self.myloss['mutual_information1'] = 0
        self.myloss['mmd_loss'] = 0
        self.myloss['negative_ll'] = 0
        self.myloss['nll_ppl'] = 0
        self.myloss['sum_log_j'] = 0
        self.myloss['end_time'] = 0

    def cal_loss(self,size,reloss,kld,mi,mmd,nll,sum_log_j):
        self.myloss['re_loss'] += reloss / size
        self.myloss['kl_divergence'] += kld / size
        self.myloss['mutual_information1'] += mi / size
        self.myloss['mmd_loss'] += mmd / size
        self.myloss['negative_ll'] += nll / size

        self.myloss['sum_log_j'] += sum_log_j / size

    def train(self, data_iter, epoch, train=True):
        start_time = time.time()
        if train is True:
            self.model.train()
        else:
            self.model.eval()
        self.init_loss()
        seq_words = 0

        size =  min(len(data_iter.data()), self.config.epoch_size * self.config.batch_size)
        for batch_id, batch in enumerate(data_iter, 1):
            if batch_id == 155:
                print("1")
            if batch_id == self.config.epoch_size:
                break
            texts, lengths = batch.text
            batch_size = texts.size(0)
            inputs = texts[:, :-1].clone()
            targets = texts[:, 1:].clone()

            # kld_weight = weight_schedule(self.config.epoch_size * ( epoch - 1 ) + batch_id)
            kld_weight = kl_anneal_function(epoch*len(data_iter) + batch_id, 0.0025, 6000 )
            # if epoch > 20:
            #     kld_weight = 0
            q_z, p_z, z, outputs, sum_log_jacobian, penalty, z0 = self.model(inputs, lengths-1)
            reloss = recon_loss(outputs, targets, pad_id, id='entropy')
            kld = total_kld(q_z, p_z)
            mi_z= mutual_info(q_z, p_z, z0)
            nll = compute_nll(q_z, p_z, z, z0, sum_log_jacobian, reloss)
            mmd = torch.zeros(1).to(z.device)
            mmd = compute_mmd(p_z, q_z)
            # loss = (reloss + kld_weight * kld) / batch_size + (2.5 - kld_weight) * mmd
            loss = (reloss + kld_weight * kld) / batch_size 
            # loss = reloss
            if train is True:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.steps += 1
            seq_words += torch.sum(lengths-1).item()
            self.cal_loss(size,reloss.item(),kld.item(),mi_z.item() * batch_size, mmd.item() * batch_size,nll.item() * batch_size,torch.sum(sum_log_jacobian).item())
            split = 'train'
            if config.tensorboard_logging:
                writer.add_scalar("%s/ELBO"%split.upper(), loss.item(), self.steps)
                writer.add_scalar("%s/NLL_Loss"%split.upper(), nll.item(), self.steps)
                writer.add_scalar("%s/KL_Loss"%split.upper(), kld.item() / batch_size, self.steps)
                writer.add_scalar("%s/KL_Weight"%split.upper(), kld_weight, self.steps)
        nll_ppl = self.myloss['negative_ll'] * size / seq_words
        if nll_ppl > 10:
            self.myloss['nll_ppl'] = math.exp(10)
        else:
            self.myloss['nll_ppl'] = math.exp(nll_ppl)
        self.myloss['end_time'] = time.time() - start_time

    def test(self):
        # Load the best saved model
        self.load()
        self.train(self.test_iter)
        self.logger.info('| Test_loss {} |'.format(self.test_loss))

    def run(self, is_train=True):
        self.steps = 0
        if is_train:
            ##############################################################################
            # Train Model
            ##############################################################################
            try:
                # for i in range(self.config.epoch):
                while True:
                    # ~~~~~~~~~~~~~~~~~~ Train ~~~~~~~~~~~~~~~~~~
                    self.logger.info(self.train_start_message)
                    self.train(self.train_iter, self.epoch, train=True)
                    self.logger.info('| Epoch {} | Train_loss {} | \n'.format(self.epoch, self.myloss))
                    # ~~~~~~~~~~~~~~~~~~ Valid ~~~~~~~~~~~~~~~~~~
                    self.logger.info(self.valid_start_message)
                    self.train(self.valid_iter, self.epoch, train=False)
                    self.logger.info('| Epoch {} | Eval_loss {} | \n'.format(self.epoch, self.myloss))
                    generated = model.sample(10, 30, sos_id=sos_id)
                    sent_str = idx_to_word(generated, idx2word=corpus.idx2word, pad_idx=pad_id)
                    print(*sent_str, sep='\n')
                    print("\n\n\n")
                    #~~~~~~~~~~~~~~~~~~ Save ~~~~~~~~~~~~~~~~~~
                    if self.myloss['re_loss'] < self.best_eval_src_acc:
                        self.save()
                        self.best_eval_src_acc = self.myloss['re_loss']
                        # self.best_eval_tgt_acc = tgt_acc
                        self.best_epoch = self.epoch
                        self.early_stop_count = 0
                    else:
                        self.early_stop_count += 1
                        if self.early_stop_count == int(self.patience * (1/3)):
                            lr = optimizer.param_groups[0]['lr'] * self.config.lr_decay
                            print('decay learning rate to %f' % lr)
                            optimizer.param_groups[0]['lr'] = lr
                        if self.early_stop_count == int(self.patience * (2/3)):
                            lr = optimizer.param_groups[0]['lr'] * self.config.lr_decay
                            print('decay learning rate to %f' % lr)
                            optimizer.param_groups[0]['lr'] = lr
                        if self.early_stop_count == int(self.patience * (5/6)):
                            # self.optimizer.init_lr = 0.1 * self.optimizer.init_lr
                            lr = optimizer.param_groups[0]['lr'] * self.config.lr_decay * 0.1
                            print('decay learning rate to %f' % lr)
                            optimizer.param_groups[0]['lr'] = lr
                    if self.early_stop_count >= self.patience:
                        self.logger.info( '\nEarly Stopping! \nBecause %d epochs the accuracy have no improvement.' % (self.patience))
                        self.logger.info( '\nthe best model is from epoch {}'.format(self.best_epoch))
                        break

                    self.epoch += 1
                # self.save()
            except KeyboardInterrupt:
                writer.close()
                # self.save()
                self.logger.info('-' * 80 + '\nExiting from training early.')
                self.logger.info( '\nthe best model is from epoch {}'.format(self.best_epoch))
            ##############################################################################
            # Test Model
            ##############################################################################
            # Test
            # self.logger.info(self.test_start_message)
            # self.test()
            # self.train(self.epoch, train=False)

            self.logger.info(self.test_start_message)
            self.train(self.test_iter, self.epoch, train=False)
            self.logger.info('Test_loss {} | \n'.format(self.myloss))
        
            # Save results
            # self.saveResults()

        else:
            ##############################################################################
            # Test Model
            ##############################################################################
            # Test
            self.logger.info(self.test_start_message)
            self.test()
            # Save results
            # self.saveResults()

    def saveResults(self):
        results = {
            # 'corpus': corpus,
            'best_epoch': self.best_epoch,
            'best_val_bleu': self.best_val_bleu,
            'test_re_loss': self.myloss['re_loss'],
        }
        # Save results
        torch.save(results, self.config.test_result_path)
        self.logger.info("Saved results state to '{}'".format(self.config.test_result_path))

if __name__ == '__main__':
    # train_iter, valid_iter, test_iter = None, None, translate_iter

    trainer = TrainerEpoch(model=model, optimizer=optimizer, criterion=None,
                            train_iter=train_iter, valid_iter=valid_iter, test_iter=test_iter,
                            logger=logger, config=config)
    print("begin run")

    trainer.run(is_train=True)
    # trainer.run(is_train=False)


    print(1)