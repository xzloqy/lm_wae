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
from data.snlidata import Corpus, get_iterator, PAD_TOKEN, SOS_TOKEN
import warnings
torch.manual_seed(1233)
random.seed(1233)
warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
# use_cuda = False
###############################################################################
# Load Config
###############################################################################
conf_path = 'machine_translation/config/config.conf'
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

                ('network params','uniform_init','float'),
                ('network params','rnn_type','str'),
                ('network params','bidirectional','bool'),
                ('network params','num_layers','int'),
                ('network params','embed_size','int'),
                ('network params','hidden_size','int'),
                ('network params','latent_size','int'),
                ('network params','word_dropout','float'),
                ('network params','embedding_dropout','float'),
                ('network params','batch_norm','bool'),
                ('network params','reconstruction_loss_function','str'),
                ('network params','use_RN','bool'),
                ('network params','forward','bool'),
                ('network params','fuse_method','str'),

                ('training params','epoch','int'),
                ('training params','lr','float'),
                ('training params','lr_decay','float'),
                ('training params','device','int'),
                ('training params','batch_size','int'),
                ('training params','max_sequence_length','int'),
                ('training params','display_freq','int'),
                ('training params','print_freq','int'),
                # ('training params', 'anneal_function', 'str'),
                # ('training params', 'k', 'float'),
                # ('training params', 'x0', 'int'),
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
    config.src = config.src.replace(config.base_dir, '')
    # config.tgt = config.tgt.replace(config.base_dir, '')
    config.src_vocab_path = config.src_vocab_path.format(config.min_vocab_freq)
    # config.tgt_vocab_path = config.tgt_vocab_path.format(config.min_vocab_freq)
    config.saved_data = config.saved_data.format(config.src, config.max_length)

    config.fuse_method = config.fuse_method.replace(config.base_dir, '')
    # config.saved_data = config.saved_data.format(config.src, config.tgt)

    config.rnn_type = config.rnn_type.replace(config.base_dir, '').lower()
    # config.anneal_function = config.anneal_function.replace(config.base_dir, '').lower()
    config.reconstruction_loss_function = config.reconstruction_loss_function.replace(config.base_dir, '')

    assert config.rnn_type in ['rnn', 'lstm', 'gru'], 'Note:the rnn_type should choose from [rnn, lstm, gru]!'
    # assert config.anneal_function in ['logistic', 'linear'], 'Note:the anneal_function should choose from [logistic, linear]!'
    assert config.reconstruction_loss_function in ['l1', 'l2'], 'Note:the anneal_function should choose from [l1:L1Loss, l2:MSELoss]!'
    assert 0 <= config.word_dropout <= 1

    if config.use_RN:
        if config.forward:
            last = 'f'
        else:
            last = 'b'
        config.log_dir = config.log_dir.format(config.src, config.batch_size, 'with', config.fuse_method) + last + '.log'
        config.best_param_path = config.best_param_path.format(config.src, config.tgt, config.batch_size, 'with', config.fuse_method) + last + '.pt'
        config.test_result_path = config.test_result_path.format(config.src, config.tgt, config.batch_size, 'with', config.fuse_method) + last + '.txt'
        config.decode_output = config.decode_output.format(config.src, config.tgt, config.batch_size, 'with', config.fuse_method) + last +'.txt'
    else:
        config.log_dir = config.log_dir.format(config.src,  config.batch_size, 'without', '') + '.log'
        config.best_param_path = config.best_param_path.format(config.src, config.batch_size, 'without', '') + '.pt'
        config.test_result_path = config.test_result_path.format(config.src, config.batch_size, 'without', '') + '.txt'
        config.decode_output = config.decode_output.format(config.src, config.batch_size, 'without', ) + '.txt'
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

def idx_to_word(idx, idx2word, pad_idx):
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_idx in sent:
            if word_idx == pad_idx:
                break
            sent_str[i] += idx2word[word_idx] + " "
        sent_str[i] = sent_str[i].strip()
    
    return sent_str
###############################################################################
# Load Data
###############################################################################
print('Loading corpus...')
# dataset = 'trans_en'
# datadir = '~/xiangzheng/machine_translation/data/trans_en'
dataset = 'snli'
datadir = '~/xiangzheng/machine_translation/data/snli'
corpus = Corpus(
    datadir, max_vocab_size=20000,
    max_length=200
)
pad_id = corpus.word2idx[PAD_TOKEN]
sos_id = corpus.word2idx[SOS_TOKEN]
vocab_size = len(corpus.word2idx)
config.src_vocab = corpus.idx2word
# config.tgt_vocab = data.tgt_vocab['vocab']       # [14627]
config.src_vocab_size = len(corpus.idx2word)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter = get_iterator(corpus.train, config.batch_size, True,  device)
valid_iter = get_iterator(corpus.valid, config.batch_size, False, device)
test_iter  = get_iterator(corpus.test,  config.batch_size, False, device)

#############################################################################
# Prepare Model
###############################################################################
print('Preparing models...')
model = AutoEncoder(config)

# src_vocab_mask = torch.ones(config.src_vocab_size)         # [2958]
# src_vocab_mask[config.src_vocab.index('<pad>')] = 0 #xz
# src_cross_entropy_loss = nn.CrossEntropyLoss(weight=src_vocab_mask, size_average=False)

if use_cuda:
    print('Using GPU..')
    model.cuda(device=config.device)
    # src_cross_entropy_loss.cuda(device=config.device)
    # tgt_cross_entropy_loss.cuda(device=config.device)
if config.uniform_init:
    print('uniformly initialize parameters [-%f, +%f]' % (config.uniform_init, config.uniform_init))
    for p in model.parameters():
        p.data.uniform_(-config.uniform_init, config.uniform_init)
optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

# criterion = src_cross_entropy_loss

###############################################################################
# Trainer
###############################################################################
def weight_schedule(t, start=6000, end=40000):
    return max(min((t - 6000) / 40000, 1), 0)

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
        # self.eval_tgt_acc = 0

        self.eval_bleu = None
        self.test_bleu = None

        self.best_eval_bleu = 0
        self.best_eval_src_acc = 10000
        # self.best_eval_tgt_acc = 0
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
        size =  min(len(data_iter.data()), 2000 * config.batch_size)

        seq_words = 0
        mmd_loss = 0
        self.init_loss()

        for batch_id, batch in enumerate(data_iter, 1):
            if batch_id == 2000:
                break
            texts, lengths = batch.text
            batch_size = texts.size(0)
            inputs = texts[:, :-1].clone()
            targets = texts[:, 1:].clone()

            kld_weight = weight_schedule(2000 * ( epoch - 1 ) + batch_id)

            q_z, p_z, z, outputs, sum_log_jacobian, penalty, z0 = self.model(inputs, lengths-1)
            reloss = recon_loss(outputs, targets, pad_id, id='entropy')
            kld = total_kld(q_z, p_z)
            mi_z= mutual_info(q_z, p_z, z0)
            nll = compute_nll(q_z, p_z, z, z0, sum_log_jacobian, reloss)
            mmd = torch.zeros(1).to(z.device)
            mmd = compute_mmd(p_z, q_z)
            seq_words += torch.sum(lengths-1).item()
            loss = (reloss + kld_weight * kld) / batch_size + (2.5 - kld_weight) * mmd
            if train is True:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.cal_loss(size,reloss.item(),kld.item(),mi_z.item() * batch_size, mmd.item() * batch_size,nll.item() * batch_size,torch.sum(sum_log_jacobian).item())
        self.myloss['end_time'] = time.time() - start_time
        nll_ppl = self.myloss['negative_ll'] * size / seq_words
        if nll_ppl > 10:
            self.myloss['nll_ppl'] = math.exp(10)
        else:
            self.myloss['nll_ppl'] = math.exp(nll_ppl)

    def test(self):
        # Load the best saved model
        self.load()
        self.train(self.test_iter)
        self.logger.info('| Test_loss {} |'.format(self.test_loss))

    def run(self, is_train=True):

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
                self.writer.close()
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
            self.writer.close()
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


class Interactive(object):
    def __init__(self, config, vocab2index, model, parallel=True):
        self.config = config
        self.src_path = config.src_path
        # if parallel:
            # self.tgt_path = config.tgt_path
        self.vocab2index = vocab2index
        self.model = model
        self.parallel = parallel
        self.data_src = None
        # self.data_tgt = None

    def input_transpose(self, sents, pad_token):
        max_len = max(len(s) for s in sents)
        batch_size = len(sents)
        sents_t = []
        for i in range(max_len):
            sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

        return sents_t

    def read_corpus(self, file_path, source):
        data = []
        for line in open(file_path):
            if source == 'src':
                sent = line.strip('\n').split(' ')
                data.append(sent)
            # only append <s> and </s> to the target sentence
            # elif source == 'tgt':
            #     sent = line.strip('\n').split(' ')
            #     sent = ['<s>'] + sent + ['</s>']
            #     data.append(sent)
        return data

    def getIndex(self, lst, vocab2index):
        # word2index = {'src':[], 'tgt':[]}
        word2index = {'src':[]}
        for sentence in lst:
            src_index = []
            # tgt_index = []
            for word in sentence[0]:
                src_index.append(vocab2index.get(word) if word in vocab2index.keys() else vocab2index.get('<unk>'))
            # for word in sentence[1]:
            #     tgt_index.append(vocab2index.get(word) if word in vocab2index.keys() else vocab2index.get('<unk>'))
            word2index['src'].append(src_index)
            # word2index['tgt'].append(tgt_index)
        return word2index

    def to_input_variable(self, sents, vocab, cuda=False, is_test=False):
        """
        return a tensor of shape (src_sent_len, batch_size)
        """
        sents_var = {'src':[]}
        word_ids = self.getIndex(sents, vocab)

        sents_var['src'] = Variable(torch.LongTensor(self.input_transpose(word_ids['src'],vocab['<pad>'])), volatile=is_test, requires_grad=False)
        # sents_var['tgt'] = Variable(torch.LongTensor(self.input_transpose(word_ids['tgt'],vocab['<pad>'])), volatile=is_test, requires_grad=False)
        if cuda:
            sents_var = sents_var.cuda()
        return sents_var

    def getModelInput(self):
        self.data_src = self.read_corpus(self.src_path, 'src')
        # self.data_tgt = self.read_corpus(self.tgt_path, 'tgt')
        data = list(zip(self.data_src))
        data_var = self.to_input_variable(data, self.vocab2index)
        return data_var

    def predict(self):
        hypotheses = []
        references = []
        data_var = self.getModelInput()
        with torch.no_grad():
            for step, inputs in enumerate(data_var):
                self.model.zero_grad()
                src_batch = inputs.src
                tgt_batch = inputs.tgt  # leave out the last <EOS> in target
                out = self.model(src_batch, tgt_batch)  # [4,5,23000]
                preds = out.max(2)[1]
                for i in range(tgt_batch.size(0)):
                    pred_idxs = [idx for idx in preds[i] if
                                 idx not in (self.vocab2index.index('<pad>'), self.vocab2index.index('<s>'), self.vocab2index.index('</s>'))]
                    pred_line = [self.vocab2index[idx] for idx in pred_idxs]
                    true_idxs = [idx for idx in tgt_batch[i] if
                                 idx not in (self.vocab2index.index('<pad>'), self.vocab2index.index('<s>'), self.vocab2index.index('</s>'))]
                    true_line = [self.vocab2index[idx] for idx in true_idxs]

                    hypotheses.append(pred_line)
                    references.append(true_line)

                    if step % self.config.print_freq == 0 and i % 8 == 0:
                        print('true:{}'.format(' '.join(true_line)))
                        print('pred:{} \n'.format(' '.join(pred_line)))

            bleu = get_bleu(references=references, hypotheses=hypotheses)
            w_acc = get_acc(references=references, hypotheses=hypotheses, acc_type='word_acc')
            s_acc = get_acc(references=references, hypotheses=hypotheses, acc_type='sent_acc')
            print('#'*30,'\nword_acc:',w_acc,'\nsent_acc:',s_acc,'\nbleu:',bleu,'\n','#'*30,)

if __name__ == '__main__':
    # train_iter, valid_iter, test_iter = None, None, translate_iter

    trainer = TrainerEpoch(model=model, optimizer=optimizer, criterion=None,
                            train_iter=train_iter, valid_iter=valid_iter, test_iter=test_iter,
                            logger=logger, config=config)
    print("begin run")

    trainer.run(is_train=True)
    # trainer.run(is_train=False)


    print(1)