import os
import math
from re import X
import sys
import torch.sparse as sp

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim
import time

from transformer.batch import subsequent_mask
from transformer.noam_opt import NoamOpt
from transformer.decoder import Decoder
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.pointerwise_feedforward import PointerwiseFeedforward
from transformer.encoder_decoder import EncoderDecoder
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
from transformer.batch import subsequent_mask

import scipy.io
import copy
import math

from torch.utils.tensorboard import SummaryWriter


class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        self.emb = nn.Conv1d(
            64,
            256,
            1,
            bias=True)

        self.demb = nn.Conv1d(
            256,
            64,
            1,
            bias=True)

    def forward(self, x, A):

        assert A.size(0) == self.kernel_size # 8
        x = self.conv(x)

        newA = A
        
        # embA = self.emb(A)
        # dembA = self.demb(embA)
        # level_weight = F.softmax(dembA, dim=1)
        # # newA = level_weight * A
        
        # threshold = 0.5  # Adjust this threshold as needed
        # sparse_level_weight = (level_weight > threshold).float()
        # newA = sparse_level_weight * A
        
        # # Apply graph convolution
        x = torch.einsum('nctv,tvw->nctw', (x, newA))

        return x.contiguous(), newA
    

class st_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])  # 3 --> 5
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res.contiguous()
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

class SGTN(nn.Module):
    def __init__(self,n_sstgcn=5,n_txpcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3, 
                 emb_size=512, fw=128,heads=8,layers=6,dropout=0.1,
                 checkpoint_dir='../../../scratch/experiment/'):
        super(SGTN,self).__init__()

        
        self.n_sstgcn= n_sstgcn
        self.n_txpcnn = n_txpcnn
                
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_sstgcn):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len))) # [1, 2, 8, num_person]

           
        self.target_embedding = nn.Sequential(
            nn.Conv2d(
                input_feat,
                output_feat,
                kernel_size=1,
                stride=(1, 1)),
            # nn.BatchNorm2d(output_feat), # bias=bias
        )

        c = copy.deepcopy
        attn = MultiHeadAttention(heads, emb_size)
        ff = PointerwiseFeedforward(emb_size, fw, dropout)
        position = PositionalEncoding(emb_size, dropout)

        self.model = EncoderDecoder(
            Encoder(EncoderLayer(emb_size, c(attn), c(ff), dropout), layers),
            Decoder(DecoderLayer(emb_size, c(attn), c(attn), c(ff), dropout), layers),
            nn.Sequential(LinearEmbedding(output_feat,emb_size), c(position)),
            nn.Sequential(LinearEmbedding(output_feat,emb_size), c(position)),
            Generator(emb_size, output_feat)) # 512-->5


        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.feat = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())


    def forward(self,v, a, v_t, return_feat=False):
        # [1, 2, 8, num_person]  # [8, num_person, num_person] 

        for k in range(self.n_sstgcn):
            v, a = self.st_gcns[k](v, a)


        device = v_t.device
        # [1, 2, 7, num_person]  # [2, num_person, num_person]  # [1, 3, 12, num_person] #[num_person]
 
        # if (v_t.size()[1]==2):
        v_t = self.target_embedding(v_t) # [1, 5, 12, num_person]

        src = v.reshape(-1,v.shape[2],v.shape[1]) # bs*num,8,5
        trg = v_t.reshape(-1,v_t.shape[2],v_t.shape[1]) # bs*num,12,5

        src_att = torch.ones((src.shape[0], 1,src.shape[1])).to(device) # [bs*num, 1, 8]
        trg_att=subsequent_mask(trg.shape[1]).repeat(trg.shape[0],1,1).to(device) # [bs*num, 12, 12]

        pred=self.model.generator(self.model(src, trg, src_att, trg_att)) # bs*num,12,5

        if return_feat:
            feat = self.feat(pred.reshape(-1,pred.shape[1],pred.shape[2],v_t.shape[3]))  # [1, 12, 5, num_person]
            feat = feat.reshape(feat.size()[3],-1)

        pred = pred.reshape(-1, pred.shape[2], pred.shape[1], v_t.shape[3]) # [1, 5, 12, num_person]

        if return_feat:
            return pred,a, feat
        else:
            return pred,a



        


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model) #2/3-->512
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_size):
        # 512 --> 3
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, out_size)

    def forward(self, x):
        return self.proj(x)
