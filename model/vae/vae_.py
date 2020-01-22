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
        self.max_sequence_length = config.max_sequence_length
        self.batch_size = config.batch_size

        self.sos_idx = config.src_vocab.index('<s>')   # 2
        self.eos_idx = config.src_vocab.index('</s>')  # 3
        self.pad_idx = config.src_vocab.index('<pad>') # 0
        self.unk_idx = config.src_vocab.index('<unk>') # 1

        self.latent_size = config.latent_size

        self.rnn_type = config.rnn_type
        self.bidirectional = config.bidirectional
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size

        self.src_embedding = nn.Embedding(config.src_vocab_size, config.embed_size)
        # self.tgt_embedding = nn.Embedding(config.tgt_vocab_size, config.embed_size)
        self.word_dropout_rate = config.word_dropout
        self.src_embedding_dropout = nn.Dropout(p=config.embedding_dropout)
        # self.tgt_embedding_dropout = nn.Dropout(p=config.embedding_dropout)

        self.embed = nn.Embedding(config.src_vocab_size, config.embed_size)
        self.encoder = Encoder(config.embed_size, config.hidden_size, config.latent_size, config.word_dropout, config.rnn_type)
        self.decoder = Decoder(config.embed_size, config.hidden_size, config.latent_size, config.word_dropout, config.rnn_type)
        self.fc = nn.Linear(config.latent_size, config.hidden_size * 2)   # used to map latent space to hidden
        self.fcout = nn.Linear(config.hidden_size, config.src_vocab_size)  # output layer

        if config.rnn_type == 'rnn':
            rnn = nn.RNN
        elif config.rnn_type == 'gru':
            rnn = nn.GRU
        elif config.rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise ValueError()

        self.src_encoder_rnn = rnn(config.embed_size, config.hidden_size, num_layers=config.num_layers, bidirectional=self.bidirectional, batch_first=True)
        # self.tgt_encoder_rnn = rnn(config.embed_size, config.hidden_size, num_layers=config.num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.src_decoder_rnn = rnn(config.embed_size, config.hidden_size, num_layers=config.num_layers, bidirectional=self.bidirectional, batch_first=True)
        # self.tgt_decoder_rnn = rnn(config.embed_size, config.hidden_size, num_layers=config.num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if config.bidirectional else 1) * config.num_layers

        self.src_hidden2mean = nn.Linear(config.hidden_size * self.hidden_factor, config.latent_size)
        self.src_hidden2logv = nn.Linear(config.hidden_size * self.hidden_factor, config.latent_size)

        # self.tgt_hidden2mean = nn.Linear(config.hidden_size * self.hidden_factor, config.latent_size)
        # self.tgt_hidden2logv = nn.Linear(config.hidden_size * self.hidden_factor, config.latent_size)

        self.batch_norm = config.batch_norm
        self.src_batch_norm_mean = nn.BatchNorm1d(config.latent_size)
        self.src_batch_norm_logv = nn.BatchNorm1d(config.latent_size)
        # self.tgt_batch_norm_mean = nn.BatchNorm1d(config.latent_size)
        # self.tgt_batch_norm_logv = nn.BatchNorm1d(config.latent_size)

        if config.reconstruction_loss_function == 'l2':
            self.self_reconstruction_criterion = nn.MSELoss(size_average=False)
        elif config.reconstruction_loss_function == 'l1':
            self.self_reconstruction_criterion = nn.L1Loss(size_average=False)

        # self.cross_reconstruction_criterion = F.cross_entropy

        self.src_latent2hidden = nn.Linear(config.latent_size, config.hidden_size * self.hidden_factor)
        # self.src_hidden2embed = nn.Linear(config.hidden_size * self.hidden_factor, config.embed_size)
        # self.src_hidden2embed = nn.Linear(config.hidden_size * self.hidden_factor, config.embed_size)
        self.src_vocab2embed = nn.Linear(config.src_vocab_size, config.embed_size)
        self.src_outputs2vocab = nn.Linear(config.hidden_size * (2 if config.bidirectional else 1), config.src_vocab_size)

        # self.tgt_latent2hidden = nn.Linear(config.latent_size, config.hidden_size * self.hidden_factor)
        # self.tgt_hidden2embed = nn.Linear(config.hidden_size * self.hidden_factor, config.embed_size)
        # self.tgt_vocab2embed = nn.Linear(config.tgt_vocab_size, config.embed_size)
        # self.tgt_outputs2vocab = nn.Linear(config.hidden_size * (2 if config.bidirectional else 1), config.tgt_vocab_size)

    def reparameterization(self, hidden, type='src'):
        # REPARAMETERIZATION
        # h = torch.cat(hidden, 2).squeeze(0)
        h = hidden
        if type == 'src':
            mean = self.src_hidden2mean(h)
            logv = self.src_hidden2logv(h)
            if self.batch_norm:
                mean, logv = self.src_batch_norm_mean(mean), self.src_batch_norm_logv(logv)
        elif type == 'tgt':
            mean = self.tgt_hidden2mean(h)
            logv = self.tgt_hidden2logv(h)
            if self.batch_norm:
                mean, logv = self.tgt_batch_norm_mean(mean), self.tgt_batch_norm_logv(logv)
        std = torch.exp(0.5 * logv)

        
        z = Variable(torch.randn([hidden.size(0), self.latent_size]).cuda(device=self.config.device))
        z = z * std + mean

        p_z = Normal(torch.zeros((h.size(0), self.latent_size), device=h.device),
                      (0.5 * torch.zeros((h.size(0), self.latent_size), device=h.device)).exp())
        q_z = Normal(mean, (0.5 * logv).exp())
        return mean, logv, z ,q_z ,p_z

    def reEncoder(self, dec_outputs, type='src'):
        if type == 'src':
            dec_outputs = self.src_vocab2embed(dec_outputs)
            _, hidden = self.src_encoder_rnn(dec_outputs)  # [2,4,512]
        elif type == 'tgt':
            dec_outputs = self.tgt_vocab2embed(dec_outputs)
            _, hidden = self.tgt_encoder_rnn(dec_outputs)  # [2,4,512]
        else:
            raise ValueError()
        if self.bidirectional or self.num_layers > 1:  # [4, 1024]
            # flatten hidden state
            hidden = hidden.view(dec_outputs.size(0), self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        return hidden

    def srcEncoder(self, input_sequence, sorted_lengths):
        input_embedding = self.src_embedding(input_sequence)            # [4,51,512]
        packed_input = pack_padded_sequence(input_embedding, sorted_lengths.tolist(), batch_first=True)
        _, hidden = self.src_encoder_rnn(packed_input)                  # [2,4,512]
        if self.bidirectional or self.num_layers > 1:               # [4, 1024]
            # flatten hidden state
            hidden = hidden.view(input_embedding.size(0), self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        return hidden

    def tgtEncoder(self, input_sequence, sorted_lengths):
        input_embedding = self.src_embedding(input_sequence)            # [4,51,512]
        packed_input = pack_padded_sequence(input_embedding, sorted_lengths.tolist(), batch_first=True)
        _, hidden = self.src_encoder_rnn(packed_input)                  # [2,4,512]
        if self.bidirectional or self.num_layers > 1:               # [4, 1024]
            # flatten hidden state
            hidden = hidden.view(input_embedding.size(0), self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()
        return hidden

    def srcDecoder(self, input_sequence, sorted_lengths, sorted_idx, z):
        hidden = self.src_latent2hidden(z)
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, input_sequence.size(0), self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)
        # decoder input
        input_embedding = self.src_embedding(input_sequence)
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size()).cuda(device=self.config.device)
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.src_embedding(decoder_input_sequence)
        input_embedding = self.src_embedding_dropout(input_embedding)
        packed_input = pack_padded_sequence(input_embedding, sorted_lengths.tolist(), batch_first=True)
        # decoder forward pass
        outputs, _ = self.src_decoder_rnn(packed_input, hidden)
        # process outputs
        padded_outputs = pad_packed_sequence(outputs, batch_first=True)[0].contiguous()
        # process outputs
        _, reversed_idx = torch.sort(sorted_idx)
        dec_outputs = padded_outputs[reversed_idx]
        # project outputs to vocab
        logp = F.log_softmax(self.src_outputs2vocab(dec_outputs.view(-1, dec_outputs.size(2))), dim=-1)
        logp = logp.view(dec_outputs.size(0), dec_outputs.size(1), self.src_embedding.num_embeddings)
        return logp

    def tgtDecoder(self, input_sequence, sorted_lengths, sorted_idx, z):
        hidden = self.tgt_latent2hidden(z)
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, input_sequence.size(0), self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)
        # decoder input
        input_embedding = self.tgt_embedding(input_sequence)
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size()).cuda(device=self.config.device)
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.tgt_embedding(decoder_input_sequence)
        input_embedding = self.tgt_embedding_dropout(input_embedding)
        packed_input = pack_padded_sequence(input_embedding, sorted_lengths.tolist(), batch_first=True)
        # decoder forward pass
        outputs, _ = self.tgt_decoder_rnn(packed_input, hidden)
        # process outputs
        padded_outputs = pad_packed_sequence(outputs, batch_first=True)[0].contiguous()
        # process outputs
        _, reversed_idx = torch.sort(sorted_idx)
        dec_outputs = padded_outputs[reversed_idx]
        # project outputs to vocab
        logp = F.log_softmax(self.tgt_outputs2vocab(dec_outputs.view(-1, dec_outputs.size(2))), dim=-1)
        logp = logp.view(dec_outputs.size(0), dec_outputs.size(1), self.tgt_embedding.num_embeddings)
        return logp


    # def weight_schedule(self, t, start=6000, end=40000):
    #     return max(min((t - 6000) / 40000, 1), 0)

    def forward(self, src_input_sequence, src_length):
        src_sorted_lengths, src_sorted_idx = torch.sort(src_length, descending=True)
        src_sorted_input_sequence = src_input_sequence[src_sorted_idx]

        batch_size = src_input_sequence.size(0)
        enc_embeds = self.embed(src_sorted_input_sequence)

        # src_hidden1 = self.srcEncoder(src_sorted_input_sequence, src_sorted_lengths)         # [4, 1024]    embedding
        hn, q_z, p_z = self.encoder(enc_embeds, src_length)
        # src_mean, src_logv, src_z, q_z, p_z = self.reparameterization(src_hidden1, 'src')       # [4, 16]
        # tgt_mean, tgt_logv, tgt_z = self.reparameterization(tgt_hidden1, 'tgt')       # [4, 16]
        ##############################################
        # DECODER           Reconstruct inputs
        ##############################################
        # src_logp = self.srcDecoder(src_sorted_input_sequence, src_sorted_lengths, src_sorted_idx, src_z)              # [4, 51, 53455]
        # tgt_logp = self.tgtDecoder(tgt_sorted_input_sequence, tgt_sorted_lengths, tgt_sorted_idx, tgt_z)              # [4, 48, 33581]
        # cross_reconstruction_loss = self.cross_reconstruction_criterion(src_logp.view(-1, self.config.src_vocab_size), src_input_sequence.view(-1), reduction='sum', ignore_index=self.pad_idx) + \
        #                             self.cross_reconstruction_criterion(tgt_logp.view(-1, self.config.tgt_vocab_size), tgt_input_sequence.view(-1), reduction='sum', ignore_index=self.pad_idx)
        ##############################################
        # RE ENCODER
        ##############################################
        # src_hidden2 = self.reEncoder(src_logp, type='src')                         # [4, 1024]
        # tgt_hidden2 = self.reEncoder(tgt_logp, type='tgt')                         # [4, 1024]
        # self_reconstruction_loss = self.self_reconstruction_criterion(src_hidden2, src_hidden1)
                                #    self.self_reconstruction_criterion(tgt_hidden2, tgt_hidden1)
        ##############################################
        # Cross Alignment     Cross Reconstruction Loss
        # 计算分别通过编码器E_1、E_2后得到的z1、z2交叉通过解码器D_1、D_2得到的x1'、x2'的欧式距离
        # loss_CA
        ##############################################
        # src_from_tgt = self.srcDecoder(src_input_sequence, src_sorted_lengths, src_sorted_idx, tgt_z)               # [4, 51, 53455]
        # tgt_from_src = self.tgtDecoder(tgt_input_sequence, tgt_sorted_lengths, tgt_sorted_idx, src_z)
            # [4, 48, 33581]
        # cross_reconstruction_loss = self.cross_reconstruction_criterion(src_from_tgt.view(-1, self.config.src_vocab_size), src_input_sequence.view(-1), reduction='sum', ignore_index=self.pad_idx) + \
        #                             self.cross_reconstruction_criterion(tgt_from_src.view(-1, self.config.tgt_vocab_size), tgt_input_sequence.view(-1), reduction='sum', ignore_index=self.pad_idx)
        # ##############################################
        # # KL-Divergence     让q(z)与p(z|X)近可能的相似      (<0)
        # ##############################################
        # KLD = (0.5 * torch.sum(1 + src_logv - src_mean.pow(2) - tgt_logv.exp())) \
        #       + (0.5 * torch.sum(1 + tgt_logv - tgt_mean.pow(2) - src_logv.exp()))
        ##############################################
        # Distribution Alignment
        # 计算z_1、z_2的概率分布的相似程度，采用的是Wasserstein距离
        # 分布对齐损失就是所有组合情况的Wasserstein距离之和
        # loss_DA
        ##############################################
        # distance = torch.sqrt(torch.sum((tgt_mean - src_mean) ** 2, dim=1) + \
        #                       torch.sum((torch.sqrt(tgt_logv .exp()) -
        #                                  torch.sqrt(src_logv.exp())) ** 2, dim=1))
        # distance = distance.sum()
        # 需要加一个wasserstein距离 xzloqy
        if self.training:
            z = q_z.rsample()
        else:
            z = q_z.mean
        z0 = z
        sum_log_jacobian = torch.zeros_like(z)
        sum_penalty = torch.zeros(1).to(z.device)

        init_hidden = torch.tanh(self.fc(z)).unsqueeze(0)
        init_hidden = [hn.contiguous() for hn in torch.chunk(init_hidden, 2, 2)]
        dec_embeds = self.embed(src_sorted_input_sequence)
        outputs, _ = self.decoder(dec_embeds, src_length, init_hidden=init_hidden)
        outputs = self.fcout(outputs)

        # reconstruction_loss = self_reconstruction_loss + cross_reconstruction_loss
        return q_z, p_z, z, outputs, sum_log_jacobian, sum_penalty, z0

    def generate(self, z, type):
        batch_size = z.size(0)
        generated = torch.zeros((batch_size, self.config.max_length), dtype=torch.long).cuda(self.config.device)
        input_sequence = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long).cuda(self.config.device)
        if type == 'src':
            hidden = self.src_latent2hidden(z)
            if self.bidirectional or self.num_layers > 1:
                hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
            else:
                hidden = hidden.unsqueeze(0)
            for k in range(self.config.max_length):
                input_embedding = self.src_embedding(input_sequence)
                if self.word_dropout_rate > 0:
                    # randomly replace decoder input with <unk>
                    prob = torch.rand(input_sequence.size()).cuda(device=self.config.device)
                    prob[(input_sequence - self.sos_idx) * (input_sequence - self.pad_idx) == 0] = 1
                    decoder_input_sequence = input_sequence.clone()
                    decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
                    input_embedding = self.src_embedding(decoder_input_sequence)
                input_embedding = self.src_embedding_dropout(input_embedding)
                dec_outputs, hidden = self.src_decoder_rnn(input_embedding, hidden)
                outputs = F.log_softmax(self.src_outputs2vocab(dec_outputs.view(-1, dec_outputs.size(2))), dim=-1).view(dec_outputs.size(0), dec_outputs.size(1), self.src_embedding.num_embeddings)
                dec_inputs = outputs.max(2)[1]
                generated[:, k] = dec_inputs[:, 0].clone()
        elif type == 'tgt':
            hidden = self.tgt_latent2hidden(z)
            if self.bidirectional or self.num_layers > 1:
                hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
            else:
                hidden = hidden.unsqueeze(0)
            for k in range(self.config.max_length):
                input_embedding = self.tgt_embedding(input_sequence)
                if self.word_dropout_rate > 0:
                    # randomly replace decoder input with <unk>
                    prob = torch.rand(input_sequence.size()).cuda(device=self.config.device)
                    prob[(input_sequence - self.sos_idx) * (input_sequence - self.pad_idx) == 0] = 1
                    decoder_input_sequence = input_sequence.clone()
                    decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
                    input_embedding = self.tgt_embedding(decoder_input_sequence)
                input_embedding = self.tgt_embedding_dropout(input_embedding)
                dec_outputs, hidden = self.tgt_decoder_rnn(input_embedding, hidden)
                outputs = F.log_softmax(self.tgt_outputs2vocab(dec_outputs.view(-1, dec_outputs.size(2))), dim=-1).view(dec_outputs.size(0), dec_outputs.size(1), self.tgt_embedding.num_embeddings)
                dec_inputs = outputs.max(2)[1]
                generated[:, k] = dec_inputs[:, 0].clone()

        return generated

    def inference(self, n=10, z=None):
        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        if z is None:
            batch_size = n
            z = Variable(torch.randn([batch_size, self.latent_size]).cuda(device=self.config.device))
        else:
            batch_size = z.size(0)

        hidden = self.src_latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=tensor()).long() # idx of still generating sequences with respect to current loop

        generations = tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t < self.config.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = Variable(torch.Tensor(batch_size).fill_(self.sos_idx).long().cuda(device=self.config.device))

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=tensor()).long()

            t += 1

        return generations, z

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
