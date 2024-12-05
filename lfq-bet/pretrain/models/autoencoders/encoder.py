# This file contains the definition of the encoder for the autoencoder model. 
# The encoder is used to encode the input sequences into a latent representation. 
# The encoder can be used to encode the input sequences into a latent representation for the autoencoder model.

import torch
import torch.nn as nn
from einops import rearrange
from pretrain.models.autoencoders.nets.mlp import MLP
from pretrain.models.autoencoders.nets.cnn import CausalConv1d, SpatialDownsample2x, SameConv1d
from typing import Union, Tuple
from einops import rearrange

class MLPChunkActEncoder(nn.Module):
    def __init__(self,
                input_dim: int,
                latent_dim: int,
                hidden_dim: int = 128,
                num_layers: int = 1,
                norm=True,
                act_chunk_len: int = 10,
                act="SiLU",
                name="ChunkActEncoder",
                ):
        super().__init__()
        self.act_chunk_len = act_chunk_len
        in_dim = input_dim * act_chunk_len
        self.net = MLP( in_dim=in_dim,
                        out_dim=latent_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        norm=norm,
                        act=act,
                        name=name)
    
    def forward(self, x):
        x = rearrange(x, "B (T Tc) Da -> B T (Tc Da)", Tc=self.act_chunk_len)
        x = self.net(x)
        return x  #B T/Tc D


class Causal1DCNNEncoder(nn.Module):
    def __init__(self,
                input_dim: int,
                latent_dim: int,
                kernel_size: Union[int, Tuple[int, int, int]],
                hidden_dim: int = 128,
                num_layers: int = 1,
                
                # norm=True,
                # act_chunk_len: int = 10,
                # act="ELU",
                # name="ChunkActEncoder",
                ):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module(f"conv{0}", CausalConv1d(input_dim, hidden_dim, kernel_size=kernel_size))
        self.net.add_module(f"act{0}", nn.ELU())
        for layer in range(num_layers - 1):
            self.net.add_module(f"conv{layer + 1}", SameConv1d(hidden_dim, hidden_dim, kernel_size=kernel_size))
            self.net.add_module(f"act{layer + 1}", nn.ELU())
        self.net.add_module(f"conv{layer + 2}", SameConv1d(hidden_dim, latent_dim, kernel_size=kernel_size))
        self.net.add_module(f"act{layer + 2}", nn.ELU())
     
    def forward(self, x):
        x = rearrange(x, "B T D -> B D T")
        x = self.net(x)
        x = rearrange(x, "B D T -> B T D")
        return x  