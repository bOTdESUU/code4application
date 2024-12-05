import torch
from torch import nn
# from pretrain.models.autoencoders.nets.mlp import MLP, VQBeTEncoderMLP
from einops import rearrange, repeat, reduce, pack, unpack
from vector_quantize_pytorch import ResidualVQ



class BaseVAE(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 latent_layer: nn.Module,
                 decoder: nn.Module,
                 action_chunk_len: int = 10,
                 ):
        super().__init__()
        self.encoder = encoder
        self.latent_layer = latent_layer
        self.decoder = decoder

    def model_step(self, x):
        x = self.encoder(x)
        (x, index, aux_loss), loss_breakdown = self.latent_layer(x, return_loss_breakdown=True)
        x = self.decoder(x)
        return x, index, aux_loss, loss_breakdown

    def forward(self, x):
        x = self.encoder(x)
        x, _, _ = self.latent_layer(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        _, index, _ = self.latent_layer(x)
        return index

    def decode(self, index):
        x = self.latent_layer.indices_to_codes(index)
        x = self.decoder(x)
        return x



class BaseVQVAE(BaseVAE):

    def indices_to_codes(self, indices):
        return self.latent_layer.indices_to_codes(indices)
        


# class VQBeTActVAE(torch.nn.Module):
#     def __init__(
#         self,
#         obs_dim=60,
#         input_dim_h=10,  # length of action chunk
#         input_dim_w=9,  # action dim
#         n_latent_dims=512,
#         vqvae_n_embed=32,
#         vqvae_groups=4,
#         eval=True,
#         device="cuda",
#         load_dir=None,
#         encoder_loss_multiplier=1.0,
#         act_scale=1.0,
#     ):
#         self.n_latent_dims = n_latent_dims
#         self.input_dim_h = input_dim_h
#         self.input_dim_w = input_dim_w
#         self.rep_dim = self.n_latent_dims
#         self.vqvae_n_embed = vqvae_n_embed
#         self.vqvae_lr = 1e-3
#         self.vqvae_groups = vqvae_groups
#         self.device = device
#         self.encoder_loss_multiplier = encoder_loss_multiplier
#         self.act_scale = act_scale

#         discrete_cfg = {"groups": self.vqvae_groups, "n_embed": self.vqvae_n_embed}

#         self.vq_layer = ResidualVQ(
#             dim=self.n_latent_dims,
#             num_quantizers=discrete_cfg["groups"],
#             codebook_size=self.vqvae_n_embed,
#         ).to(self.device)
#         self.embedding_dim = self.n_latent_dims

#         self.vq_layer.device = device

#         if self.input_dim_h == 1:
#             self.encoder = VQBeTEncoderMLP(
#                 input_dim=input_dim_w, output_dim=n_latent_dims
#             ).to(self.device)
#             self.decoder = VQBeTEncoderMLP(
#                 input_dim=n_latent_dims, output_dim=input_dim_w
#             ).to(self.device)
#         else:
#             self.encoder = VQBeTEncoderMLP(
#                 input_dim=input_dim_w * self.input_dim_h, output_dim=n_latent_dims
#             ).to(self.device)
#             self.decoder = VQBeTEncoderMLP(
#                 input_dim=n_latent_dims, output_dim=input_dim_w * self.input_dim_h
#             ).to(self.device)

#         params = (
#             list(self.encoder.parameters())
#             + list(self.decoder.parameters())
#             + list(self.vq_layer.parameters())
#         )
#         self.vqvae_optimizer = torch.optim.Adam(
#             params, lr=self.vqvae_lr, weight_decay=0.0001
#         )

#         if load_dir is not None:
#             try:
#                 state_dict = torch.load(load_dir)
#             except RuntimeError:
#                 state_dict = torch.load(load_dir, map_location=torch.device("cpu"))
#             self.load_state_dict(state_dict)

#         if eval:
#             self.vq_layer.eval()
#         else:
#             self.vq_layer.train()

#     def draw_logits_forward(self, encoding_logits):
#         z_embed = self.vq_layer.draw_logits_forward(encoding_logits)
#         return z_embed

#     def draw_code_forward(self, encoding_indices):
#         with torch.no_grad():
#             z_embed = self.vq_layer.get_codes_from_indices(encoding_indices)
#             z_embed = z_embed.sum(dim=0)
#         return z_embed

#     def get_action_from_latent(self, latent):
#         output = self.decoder(latent) * self.act_scale
#         if self.input_dim_h == 1:
#             return rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)
#         else:
#             return rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

#     def preprocess(self, state):
#         # if not torch.is_tensor(state):
#         #     state = get_tensor(state, self.device)
#         # if self.input_dim_h == 1:
#         #     state = state.squeeze(-2)  # state.squeeze(-1)
#         # else:
#         #     state = rearrange(state, "N T A -> N (T A)")
#         # return state.to(self.device)
#         return state

#     def get_code(self, state, required_recon=False):
#         state = state / self.act_scale
#         state = self.preprocess(state)
#         with torch.no_grad():
#             state_rep = self.encoder(state)
#             state_rep_shape = state_rep.shape[:-1]
#             state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
#             state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
#             state_vq = state_rep_flat.view(*state_rep_shape, -1)
#             vq_code = vq_code.view(*state_rep_shape, -1)
#             vq_loss_state = torch.sum(vq_loss_state)
#             if required_recon:
#                 recon_state = self.decoder(state_vq) * self.act_scale
#                 recon_state_ae = self.decoder(state_rep) * self.act_scale
#                 if self.input_dim_h == 1:
#                     return state_vq, vq_code, recon_state, recon_state_ae
#                 else:
#                     return (
#                         state_vq,
#                         vq_code,
#                         torch.swapaxes(recon_state, -2, -1),
#                         torch.swapaxes(recon_state_ae, -2, -1),
#                     )
#             else:
#                 # econ_from_code = self.draw_code_forward(vq_code)
#                 return state_vq, vq_code

#     def vqvae_update(self, state):
#         state = state / self.act_scale
#         state = self.preprocess(state)
#         state_rep = self.encoder(state)
#         state_rep_shape = state_rep.shape[:-1]
#         state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
#         state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
#         state_vq = state_rep_flat.view(*state_rep_shape, -1)
#         vq_code = vq_code.view(*state_rep_shape, -1)
#         vq_loss_state = torch.sum(vq_loss_state)

#         dec_out = self.decoder(state_vq)
#         encoder_loss = (state - dec_out).abs().mean()

#         rep_loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)

#         # Optimize the critic
#         self.vqvae_optimizer.zero_grad()
#         rep_loss.backward()
#         self.vqvae_optimizer.step()
#         vqvae_recon_loss = torch.nn.MSELoss()(state, dec_out)
#         return (
#             encoder_loss.clone().detach(),
#             vq_loss_state.clone().detach(),
#             vq_code,
#             vqvae_recon_loss.item(),
#         )

#     def state_dict(self):
#         return {
#             "encoder": self.encoder.state_dict(),
#             "decoder": self.decoder.state_dict(),
#             "optimizer": self.vqvae_optimizer.state_dict(),
#             "vq_embedding": self.vq_layer.state_dict(),
#         }

#     def load_state_dict(self, state_dict):
#         self.encoder.load_state_dict(state_dict["encoder"])
#         self.decoder.load_state_dict(state_dict["decoder"])
#         self.vqvae_optimizer.load_state_dict(state_dict["optimizer"])
#         self.vq_layer.load_state_dict(state_dict["vq_embedding"])
#         self.vq_layer.eval()
