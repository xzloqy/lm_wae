#!/usr/bin/env python36
#-*- coding:utf-8 -*-
# @Time    : 19-11-22 下午5:06
# @Author  : Xinxin Zhang
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim, 
                 dropout, 
                 enc_type='lstm', batch_norm=True):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)

        self.fcmu = nn.Linear(hidden_dim * 2, code_dim)
        self.fclv = nn.Linear(hidden_dim * 2, code_dim)
        self.fcvar = nn.Linear(hidden_dim * 2, 1)
        self.bnmu = nn.BatchNorm1d(code_dim)
        self.bnlv = nn.BatchNorm1d(code_dim)
        self.bn = batch_norm
        self.code_dim = code_dim

    def forward(self, inputs, lengths):
        inputs = pack(self.drop(inputs), lengths, batch_first=True, enforce_sorted=False)
        _, hn = self.rnn(inputs)
        h = torch.cat(hn, dim=2).squeeze(0)
        p_z = Normal(torch.zeros((h.size(0), self.code_dim), device=h.device),
                    (0.5 * torch.zeros((h.size(0), self.code_dim), device=h.device)).exp())
        mu, lv = self.fcmu(h), self.fclv(h)
        q_z = Normal(mu, (0.5 * lv).exp())
        if self.bn:
            mu, lv = self.bnmu(mu), self.bnlv(lv)
        return hn,q_z, p_z

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim, 
                 dropout, 
                 de_type='lstm'):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, inputs, lengths=None, init_hidden=None):
        inputs = self.drop(inputs)
        if lengths is not None:
            inputs = pack(inputs, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(inputs, init_hidden)
        if lengths is not None:
            outputs, _ = unpack(outputs, batch_first=True)
        outputs = self.drop(outputs)
        return outputs, hidden

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.input_vocab_size, config.embed_size)
        self.encoder = Encoder(config.embed_size, config.hidden_size, config.latent_size, config.word_dropout, config.rnn_type)
        self.decoder = Decoder(config.embed_size, config.hidden_size, config.latent_size, config.word_dropout, config.rnn_type)
        self.fc = nn.Linear(config.latent_size, config.hidden_size * 2)   # used to map latent space to hidden
        self.fcout = nn.Linear(config.hidden_size, config.input_vocab_size)  # output layer
        # self.src_hidden2mean = nn.Linear(config.hidden_size * 2, config.latent_size)
        # self.src_hidden2logv = nn.Linear(config.hidden_size * 2, config.latent_size)
        # self.batch_norm = config.batch_norm
        # self.src_batch_norm_mean = nn.BatchNorm1d(config.latent_size)
        # self.src_batch_norm_logv = nn.BatchNorm1d(config.latent_size)

    def forward(self, src_input_sequence, src_length):
        # src_sorted_lengths, src_sorted_idx = torch.sort(src_length, descending=True)
        # src_sorted_input_sequence = src_input_sequence[src_sorted_idx]
        batch_size = src_input_sequence.size(0)
        enc_embeds = self.embed(src_input_sequence)
        hn, q_z, p_z = self.encoder(enc_embeds, src_length)
        if self.training:
            z = q_z.rsample()
        else:
            z = q_z.mean
        z0 = z
        sum_log_jacobian = torch.zeros_like(z)
        sum_penalty = torch.zeros(1).to(z.device)
        init_hidden = torch.tanh(self.fc(z)).unsqueeze(0)
        init_hidden = [hn.contiguous() for hn in torch.chunk(init_hidden, 2, 2)]
        dec_embeds = self.embed(src_input_sequence)
        outputs, _ = self.decoder(dec_embeds, src_length, init_hidden=init_hidden)
        outputs = self.fcout(outputs)
        return q_z, p_z, z, outputs, sum_log_jacobian, sum_penalty, z0

    def generate(self, z, max_length, sos_id):
        batch_size = z.size(0)
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=z.device)
        dec_inputs = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=z.device)
        # hidden = self.z2h(z)
        init_hidden = torch.tanh(self.fc(z)).unsqueeze(0)
        init_hidden = [hn.contiguous() for hn in torch.chunk(init_hidden, 2, 2)]
        for k in range(max_length):
            # dec_emb = self.lookup(dec_inputs)
            dec_emb = self.embed(dec_inputs)
            # dec_emb = dec_inputs
            outputs, hidden = self.decoder(dec_emb, init_hidden=init_hidden)
            outputs = self.fcout(outputs)
            dec_inputs = outputs.max(2)[1]
            generated[:, k] = dec_inputs[:, 0].clone()
        return generated

    def standard_normal(self, size): # xzloqy device
        p_z = Normal(torch.zeros((size, self.config.latent_size), device=self.config.device),
                      (0.5 * torch.zeros((size, self.config.latent_size), device=self.config.device)).exp())
        return p_z

    def sample(self, num_samples, max_length, sos_id):
        prior = self.standard_normal(num_samples)
        z = prior.sample()
        return self.generate(z, max_length, sos_id)

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
