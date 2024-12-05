
import torch
import torch.nn as nn
from typing import Union, Tuple
from einops import rearrange
from pretrain.models.autoencoders.nets.mlp import MLP
from pretrain.models.autoencoders.nets.cnn import CausalConv1d, SpatialDownsample2x, SameConv1d
from einops import rearrange

class MLPChunkActDecoder(nn.Module):
    def __init__(self,
                output_dim: int,
                latent_dim: int,
                hidden_dim: int = 128,
                num_layers: int = 1,
                norm=True,
                act_chunk_len: int = 10,
                act="SiLU",
                name="ChunkActDecoder",
                ):
        super().__init__()
        self.acti_chunk_len = act_chunk_len
        out_dim = output_dim * act_chunk_len
        self.net = MLP( in_dim=latent_dim,
                        out_dim=out_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        norm=norm,
                        act=act,
                        name=name)
    
    def forward(self, x):
        x = self.net(x)
        x = rearrange(x, "B T (Tc Da) -> B (T Tc) Da", Tc=self.acti_chunk_len)
        return x  #B T/Tc D


class Causal1DCNNDecoder(nn.Module):
    def __init__(self,
                output_dim: int,
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
        self.net.add_module(f"conv{0}", CausalConv1d(latent_dim, hidden_dim, kernel_size=kernel_size))
        self.net.add_module(f"act{0}", nn.ELU())
        for layer in range(num_layers - 1):
            self.net.add_module(f"conv{layer + 1}", SameConv1d(hidden_dim, hidden_dim, kernel_size=kernel_size))
            self.net.add_module(f"act{layer + 1}", nn.ELU())
        self.net.add_module(f"conv{layer + 2}", SameConv1d(hidden_dim, output_dim, kernel_size=kernel_size))
        self.net.add_module(f"act{layer + 2}", nn.ELU())
     
    def forward(self, x):
        x = rearrange(x, "B T D -> B D T")
        x = self.net(x)
        x = rearrange(x, "B D T -> B T D")
        return x  