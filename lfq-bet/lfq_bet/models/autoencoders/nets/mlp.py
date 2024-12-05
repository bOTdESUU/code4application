import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit
from vqvae.vqvae_utils import *
import einops
from vector_quantize_pytorch import ResidualVQ

from pretrain.models.autoencoders.nets.helper import symlog, weight_init

# class VQBeTEncoderMLP(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim=16,
#         hidden_dim=128,
#         layer_num=1,
#         last_activation=None,
#     ):
#         super(VQBeTEncoderMLP, self).__init__()
#         layers = []

#         layers.append(nn.Linear(input_dim, hidden_dim))
#         layers.append(nn.ReLU())
#         for _ in range(layer_num):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())

#         self.encoder = nn.Sequential(*layers)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#         if last_activation is not None:
#             self.last_layer = last_activation
#         else:
#             self.last_layer = None
#         self.apply(weights_init_encoder)

#     def forward(self, x):
#         h = self.encoder(x)
#         state = self.fc(h)
#         if self.last_layer:
#             state = self.last_layer(state)
#         return state


class MLP(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            hidden_dim,
            num_layers : int = 1,
            act="SiLU",
            norm=True,
            # symlog_inputs=False,
            name="NoName",
        ):
            """
            Initialize the MLP (Multi-Layer Perceptron) network.

            Args:
                in_dim (int): The input dimension.
                out_dim (int): The output dimension.
                hidden_dim (int): The number of hidden units in each layer.
                num_layers (int, optional): The number of layers in the network. Defaults to 1.
                act (str, optional): The activation function to use. Defaults to "SiLU".
                norm (bool, optional): Whether to apply layer normalization. Defaults to True.
                symlog_inputs (bool, optional): Whether to use symlog inputs. Defaults to False.
                name (str, optional): The name of the network. Defaults to "NoName".
            """
            super(MLP, self).__init__()
            self._out_dim = out_dim
            act = getattr(torch.nn, act)
            # self._symlog_inputs = symlog_inputs

            self.layers = nn.Sequential()
            for i in range(num_layers):
                self.layers.add_module(
                    f"{name}_linear{i}", nn.Linear(in_dim, hidden_dim, bias=False)
                )
                if norm:
                    self.layers.add_module(
                        f"{name}_norm{i}", nn.LayerNorm(hidden_dim, eps=1e-03)
                    )
                self.layers.add_module(f"{name}_act{i}", act())
                if i == 0:
                    in_dim = hidden_dim
            self.layers.add_module(f"{name}_linear{num_layers}", nn.Linear(in_dim, self._out_dim))
            if norm:
                self.layers.add_module(f"{name}_norm{num_layers}", nn.LayerNorm(self._out_dim, eps=1e-03))
            self.layers.add_module(f"{name}_act{num_layers}", act())        

            self.layers.apply(weight_init)

    def forward(self, features, dtype=None):
        x = features
        # if self._symlog_inputs:
        #     x = symlog(x)
        out = self.layers(x)
        return out
  
