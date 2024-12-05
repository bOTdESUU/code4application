import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit
from vqvae.vqvae_utils import *
import einops
from vector_quantize_pytorch import ResidualVQ, LFQ


class BaseLatentLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError


class LFQLayer(LFQ, BaseLatentLayer):
    def __init__(
        self,
        *,
        latent_dim = None,
        codebook_size = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 0.25,
        diversity_gamma = 1.,
        straight_through_activation = nn.Identity(),
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,            # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy = 1.    # make less than 1. to only use a random fraction of the probs for per sample entropy
    ):
        super().__init__(
            dim = latent_dim,
            codebook_size = codebook_size,
            entropy_loss_weight = entropy_loss_weight,
            commitment_loss_weight = commitment_loss_weight,
            diversity_gamma = diversity_gamma,
            straight_through_activation = straight_through_activation,
            num_codebooks = num_codebooks,
            keep_num_codebooks_dim = keep_num_codebooks_dim,
            codebook_scale = codebook_scale,
            frac_per_sample_entropy = frac_per_sample_entropy
        )