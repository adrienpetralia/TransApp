#################################################################################################################
#
# @copyright : ©2023 EDF
# @author : Adrien Petralia
# @description : TransApp time series classifier
# @component: src/TransAppModel/
# @file : TransApp.py
#
#################################################################################################################

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Callable, Optional

from Encoding import PositionalEncoding1D, LearnablePositionalEncoding1D
from AttentionMask import DiagonalMask, TriangularCausalMask

# Convolutional Dilated Embedding Block Functions
class Conv1dSamePadding(nn.Conv1d):
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class ResUnit(nn.Module):       
    def __init__(self, c_in, c_out, k=8, dilation=1, stride=1, bias=True):
        super().__init__()
        
        self.layers = nn.Sequential(Conv1dSamePadding(in_channels=c_in, out_channels=c_out,
                                                      kernel_size=k, dilation=dilation, stride=stride, bias=bias),
                                    nn.GELU(),
                                    nn.BatchNorm1d(c_out)
                                    )
        if c_in > 1 and c_in!=c_out:
            self.match_residual=True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual=False
            
    def forward(self,x):
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)
            
            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))

class DilatedBlock(nn.Module):  
    def __init__(self, c_in=32, c_out=32, 
                 kernel_size=8, dilation_list=[1, 2, 4, 8]):
        super().__init__()
 
        layers = []
        for i, dilation in enumerate(dilation_list):
            if i==0:
                layers.append(ResUnit(c_in, c_out, k=kernel_size, dilation=dilation))
            else:
                layers.append(ResUnit(c_out, c_out, k=kernel_size, dilation=dilation))
        self.network = torch.nn.Sequential(*layers)
            
    def forward(self,x):
        x = self.network(x)
        return x


# Multi Head Attention and Positional Feed Forward Functions
class ScaleDotProductAttention(nn.Module):
    """
    Vanilla ScaleDotProductAttention proposed by Vaswani et al. 2017 in Attention is all you Need
    Updated : Implementation of learnable temperature and diagonal masking proposed in Vision Transformer for Small-Size Datasets
    
    Use einsum instead of matmul for faster computation.
    """
    def __init__(self, attention_dropout=0.1, output_attention=False,
                 mask_diag=False, mask_flag=False, 
                 learnable_scale=False, head_dim=None):
        super().__init__()

        if learnable_scale:
            assert head_dim is not None, f"Provide head_dim if learnable scale==True"
            self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=True)
        else:
            self.scale = None
            
        self.mask_flag = mask_flag
        self.mask_diag = mask_diag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
            
        if self.mask_diag:
            diag_mask = DiagonalMask(B, L, device=queries.device)
            scores.masked_fill_(diag_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
        
class AttentionLayer(nn.Module):
    """
    Vanilla Full Attention Layer proposed by Vaswani et al. 2017 in Attention is all you Need
    """
    def __init__(self, d_model, n_heads=8, attn_dropout=0., proj_dropout=0., 
                 att_mask_diag=False, att_mask_flag=False, learnable_scale=False, 
                 output_attention=False, d_keys=None, d_values=None):
        super().__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = ScaleDotProductAttention(attention_dropout=attn_dropout, output_attention=output_attention,
                                                        mask_diag=att_mask_diag, mask_flag=att_mask_flag,
                                                        learnable_scale=learnable_scale, head_dim=d_keys)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.Dropout = nn.Dropout(proj_dropout)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)

        out, att = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        out = self.Dropout(self.out_projection(out))

        return out, att
        
        
class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dp_rate=0., activation=F.gelu, bias1=True, bias2=True):
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim, bias=bias1)
        self.layer2 = nn.Linear(hidden_dim, dim, bias=bias2)
        self.dropout = nn.Dropout(dp_rate)
        self.activation = activation

    def forward(self, x):
        x = self.layer2(self.dropout(self.activation(self.layer1(x))))
        return x


# Encoder Block Functions
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, 
                 dp_rate=0.2, attn_dp_rate=0.2, 
                 norm='BatchNorm', prenorm=False, 
                 store_att=False, 
                 att_mask_diag=False, att_mask_flag=False, learnable_scale=False, 
                 activation="gelu", norm_eps=1e-05):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.attention_layer = AttentionLayer(d_model, n_heads=n_heads, 
                                              attn_dropout=attn_dp_rate, proj_dropout=dp_rate, 
                                              att_mask_diag=att_mask_diag, att_mask_flag=att_mask_flag, 
                                              learnable_scale=learnable_scale, 
                                              output_attention=store_att)

        self.prenorm = prenorm
        if norm=='BatchNorm':
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model, eps=norm_eps), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model, eps=norm_eps), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

        self.dropout = nn.Dropout(dp_rate)
        f_activation = F.gelu if activation == "gelu" else F.relu
        self.pffn = PositionWiseFeedForward(dim=d_model, hidden_dim=d_ff, dp_rate=dp_rate, activation=f_activation)
        
        self.store_att = store_att
        self.att = None

    def forward(self, x) -> torch.Tensor:
        # x input and output shape [batch, seq_length, d_model] to meet Transformer Convention

        # Attention Block
        if self.prenorm:
            x = self.norm1(x)
        new_x, att = self.attention_layer(x, x, x)
        if self.store_att:
            self.att = att
        x = torch.add(x, new_x)
        if not self.prenorm:
            x = self.norm1(x)

        # PFFN Block
        if self.prenorm:
            x = self.norm2(x)
        new_x = self.pffn(x)
        x = torch.add(x, self.dropout(new_x))
        if not self.prenorm:
            x = self.norm2(x)

        return x
    
# ======================= TransApp =======================#
class TransApp(nn.Module):
    def __init__(self, 
                 max_len=1024, c_in=1,
                 mode="classif",
                 n_embed_blocks=1, 
                 encoding_type="noencoding",
                 n_encoder_layers=3,
                 kernel_size=5,
                 d_model=64, pffn_ratio=2, n_head=4,
                 prenorm=True, norm="LayerNorm",
                 activation='gelu',
                 store_att=False, attn_dp_rate=0.2, head_dp_rate=0.1, dp_rate=0.2,
                 att_param={'attenc_mask_diag': True, 'attenc_mask_flag': False, 'learnable_scale_enc': False},
                 c_reconstruct=1, apply_gap=False, nb_class=2):
        super().__init__()
  
        self.c_in = c_in
        self.d_model = d_model
        self.mode = mode
        self.nb_class = nb_class
        
        #============ Dilated Conv Embedding ============#
        layers = []
        for i in range(n_embed_blocks):
            layers.append(DilatedBlock(c_in=c_in if i==0 else d_model, 
                                       c_out=d_model, kernel_size=kernel_size))
        layers.append(Transpose(1, 2))
        self.EmbedBlock = torch.nn.Sequential(*layers) 
            
        #============ Encoding ============#
        if encoding_type == 'learnable':
            self.PosEncoding = LearnablePositionalEncoding1D(d_model, max_len=max_len)
        elif encoding_type == 'fixed':
            self.PosEncoding = PositionalEncoding1D(d_model)
        elif encoding_type == 'noencoding':
            self.PosEncoding = None
        else:
            raise ValueError('Type of encoding {} unknown, only "learnable", "fixed" or "noencoding" supported.'
                             .format(encoding_type))
        
        #============ Encoder ============#
        layers = []
        for i in range(n_encoder_layers):
            layers.append(EncoderLayer(d_model, d_model * pffn_ratio, n_head, 
                                       dp_rate=dp_rate, attn_dp_rate=attn_dp_rate, 
                                       att_mask_diag=att_param['attenc_mask_diag'], 
                                       att_mask_flag=att_param['attenc_mask_flag'], 
                                       learnable_scale=att_param['learnable_scale_enc'], 
                                       store_att=store_att,  norm=norm, prenorm=prenorm, activation=activation))
        layers.append(nn.LayerNorm(d_model))
        self.EncoderBlock = torch.nn.Sequential(*layers)
        
        #============ Pretraining Head ============#
        layers = []
        layers.append(nn.Linear(d_model, c_reconstruct, bias=True))
        layers.append(nn.Dropout(head_dp_rate))
        self.PredHead = torch.nn.Sequential(*layers)
        
        #============ Classif Head ============#
        layers = []
        if apply_gap:
            layers.append(Transpose(1,2))
            layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten(start_dim=1))
        if apply_gap:
            layers.append(nn.Linear(d_model, nb_class, bias=True))
        else:
            layers.append(nn.Linear(max_len*d_model, nb_class, bias=True))
        layers.append(nn.Dropout(head_dp_rate))
        self.ClassifHead = torch.nn.Sequential(*layers)
                      
        self.initialize_weights()
        
    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def freeze_params(self, model_part, rq_grad=False):
        for name, child in model_part.named_children():
            for param in child.parameters():
                param.requires_grad = rq_grad
            self.freeze_params(child)
    
    def forward(self, x) -> torch.Tensor:
        # Dilated Conv Embedding Block
        x = self.EmbedBlock(x)
        
        # Add Pos. Encoding (if any)
        if self.PosEncoding is not None:
            x = x + self.PosEncoding(x)
            
        # Forward Encoder
        x = self.EncoderBlock(x)
        
        # Forward Head
        if self.mode=="pretraining":
            x = self.PredHead(x).permute(0, 2, 1)
        else:
            x = self.ClassifHead(x)
                      
        return x
