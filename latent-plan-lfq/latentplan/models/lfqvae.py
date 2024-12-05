from torch.autograd import Function
from latentplan.models.autoencoders import SymbolWiseTransformer
from latentplan.models.transformers import *
from latentplan.models.ein import EinLinear
import numpy as np

from vector_quantize_pytorch import LFQ
from einops import rearrange, reduce


class LFQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        K = int(K)
        D = int(D)
        self.vq_layer = LFQ(dim=D, codebook_size=K)
    
    def forward(self, z_e_x):
        # z_e_x_ = z_e_x.contiguous()
        quantized, indices, aux_loss = self.vq_layer(z_e_x)
        return quantized, indices, aux_loss
    
    @torch.no_grad() #equal to the VectorQuantization function in TAP
    def get_indices(self, z_e_x):
        quantized, indices, aux_loss = self.vq_layer(z_e_x)
        return indices
    
    def indices_to_codes(self, indices):
        return self.vq_layer.indices_to_codes(indices)


class VQStepWiseTransformer(nn.Module):
    def __init__(self, config, feature_dim):
        super().__init__()
        self.K=config.K
        self.latent_size = int(config.trajectory_embd)
        self.condition_size = config.observation_dim
        self.trajectory_input_length = config.block_size - config.transition_dim
        self.embedding_dim = config.n_embd
        self.trajectory_length = config.block_size//config.transition_dim-1
        self.block_size = config.block_size
        self.observation_dim = feature_dim
        self.action_dim = config.action_dim
        self.transition_dim = config.transition_dim
        self.latent_step = config.latent_step
        self.state_conditional = config.state_conditional
        if "masking" in config:
            self.masking = config.masking
        else:
            self.masking = "none"

        self.encoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        if "ma_update" in config and not (config.ma_update):
            raise ValueError("Not implemented")
        else:
            # we are not actually using ema
            self.codebook = LFQEmbedding(config.K, config.trajectory_embd)
            self.ma_update = False
            # self.codebook = VQEmbeddingMovingAverage(config.K, config.trajectory_embd)
            # self.ma_update = True

        if "residual" not in config :
            self.residual = True
        else:
            self.residual = config.residual
        self.decoder = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        self.pos_emb = nn.Parameter(torch.zeros(1, self.trajectory_length, config.n_embd))

        self.embed = nn.Linear(self.transition_dim, self.embedding_dim)

        self.predict = nn.Linear(self.embedding_dim, self.transition_dim)

        self.cast_embed = nn.Linear(self.embedding_dim, self.latent_size)
        self.latent_mixing = nn.Linear(self.latent_size+self.observation_dim, self.embedding_dim)
        if "bottleneck" not in config:
            self.bottleneck = "pooling"
        else:
            self.bottleneck = config.bottleneck
        if self.bottleneck == "pooling":
            self.latent_pooling = nn.MaxPool1d(self.latent_step, stride=self.latent_step)
        # elif self.bottleneck == "attention":
        #     self.latent_pooling = AsymBlock(config, self.trajectory_length // self.latent_step)
        #     self.expand = AsymBlock(config, self.trajectory_length)
        else:
            raise ValueError(f'Unknown bottleneck type {self.bottleneck}')

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)


    def encode(self, joined_inputs):
        joined_inputs = joined_inputs.to(dtype=torch.float32)
        b, t, joined_dimension = joined_inputs.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.embed(joined_inputs)

        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        ## [ B x T x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.encoder(x)
        ## [ B x T x embedding_dim ]
        if self.bottleneck == "pooling":
            x = self.latent_pooling(x.transpose(1, 2)).transpose(1, 2)
        # elif self.bottleneck == "attention":
        #     x = self.latent_pooling(x)
        else:
            raise ValueError()

        ## [ B x (T//self.latent_step) x embedding_dim ]
        x = self.cast_embed(x)
        return x

    def decode(self, latents, state):
        """
            latents: [B x (T//self.latent_step) x latent_size]
            state: [B x observation_dimension]
        """
        B, T, _ = latents.shape
        state_flat = torch.reshape(state, shape=[B, 1, -1]).repeat(1, T, 1)
        if not self.state_conditional:
            state_flat = torch.zeros_like(state_flat)
        inputs = torch.cat([state_flat, latents], dim=-1)
        inputs = self.latent_mixing(inputs)
        if self.bottleneck == "pooling":
            inputs = torch.repeat_interleave(inputs, self.latent_step, 1)
        elif self.bottleneck == "attention":
            inputs = self.expand(inputs)

        inputs = inputs + self.pos_emb[:, :inputs.shape[1]]
        x = self.decoder(inputs)
        x = self.ln_f(x)

        ## [B x T x obs_dim]
        joined_pred = self.predict(x)
        joined_pred[:, :, -1] = torch.sigmoid(joined_pred[:, :, -1])
        joined_pred[:, :, :self.observation_dim] += torch.reshape(state, shape=[B, 1, -1])
        return joined_pred

    def forward(self, joined_inputs, state):
        trajectory_feature = self.encode(joined_inputs)

        # latents_st, latents = self.codebook.straight_through(trajectory_feature)
        quantized, indices, aux_loss = self.codebook(trajectory_feature)
        if self.bottleneck == "attention":
            raise ValueError("Not implemented")
            if self.masking == "uniform":
                mask = torch.ones(quantized.shape[0], quantized.shape[1], 1).to(quantized.device)
                mask_index = np.random.randint(0, quantized.shape[1], size=[quantized.shape[0]])
                for i, start in enumerate(mask_index):
                    mask[i, -start:, 0] = 0
                quantized = quantized * mask
                latents = latents * mask
                trajectory_feature = trajectory_feature * mask
            elif self.masking == "none":
                pass
            else:
                raise ValueError(f"Unknown masking type {self.masking}")
        joined_pred = self.decode(quantized, state)
        return joined_pred, aux_loss, trajectory_feature


class VQContinuousVAE(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem (+1 for stop token)
        self.model = VQStepWiseTransformer(config, config.observation_dim)
        self.trajectory_embd = config.trajectory_embd
        self.vocab_size = config.vocab_size
        self.stop_token = config.vocab_size * config.transition_dim
        self.block_size = config.block_size
        self.observation_dim = config.observation_dim
        if "masking" in config:
            self.masking = config.masking
        else:
            self.masking = "none"

        self.action_dim = config.action_dim
        self.trajectory_length = config.block_size//config.transition_dim-1
        self.transition_dim = config.transition_dim
        self.action_weight = config.action_weight
        self.reward_weight = config.reward_weight
        self.value_weight = config.value_weight
        self.position_weight = config.position_weight
        self.first_action_weight = config.first_action_weight
        self.sum_reward_weight = config.sum_reward_weight
        self.last_value_weight = config.last_value_weight
        self.latent_step = config.latent_step

        self.padding_vector = torch.zeros(self.transition_dim-1)
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def set_padding_vector(self, padding):
        self.padding_vector = padding

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EinLinear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        if isinstance(self.model, SymbolWiseTransformer) or isinstance(self.model, VQStepWiseTransformer):
            no_decay.add('model.pos_emb')
            if self.model.bottleneck == "attention":
                no_decay.add('model.latent_pooling.query')
                no_decay.add('model.expand.query')
                no_decay.add('model.latent_pooling.attention.in_proj_weight')
                no_decay.add('model.expand.attention.in_proj_weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    @torch.no_grad()
    def encode(self, joined_inputs, terminals):
        b, t, joined_dimension = joined_inputs.size()
        padded = torch.tensor(self.padding_vector, dtype=torch.float32,
                              device=joined_inputs.device).repeat(b, t, 1)
        terminal_mask = torch.clone(1 - terminals).repeat(1, 1, joined_inputs.shape[-1])
        joined_inputs = joined_inputs*terminal_mask+(1-terminal_mask)*padded

        trajectory_feature = self.model.encode(torch.cat([joined_inputs, terminals], dim=2))
        if self.model.ma_update:
            # indices = vq(trajectory_feature, self.model.codebook.embedding)
            indices = self.model.codebook.get_indices(trajectory_feature)
        else:
            # indices = vq(trajectory_feature, self.model.codebook.embedding.weight)
            indices = self.model.codebook.get_indices(trajectory_feature)
        return indices

    def decode(self, latent, state):
        return self.model.decode(latent, state)

    def decode_from_indices(self, indices, state):
        B, T = indices.shape
        if self.model.ma_update:
            # latent = torch.index_select(self.model.codebook.embedding, dim=0, index=indices.flatten()).reshape([B, T, -1])
            latent = quantized = self.model.codebook.indices_to_codes(indices)
        else:
            # latent = torch.index_select(self.model.codebook.embedding.weight, dim=0, index=indices.flatten()).reshape(
                # [B, T, -1])
            latent = quantized = self.model.codebook.indices_to_codes(indices)

        if self.model.bottleneck == "attention":
            # latent = torch.concat([latent, torch.zeros([B, self.trajectory_length//self.latent_step, latent.shape[2]],
            #                                            device=latent.device)],
            #                       dim=1)
            raise ValueError("Not implemented")
        state = state[:,None,:]
        return self.model.decode(latent, state.repeat(latent.shape[0], 1, 1))

    def forward(self, joined_inputs, targets=None, mask=None, terminals=None, returnx=False):
        """
            joined_inputs : [ B x T x joined_dimension]
            values : [ B x 1 x 1 ]
        """

        joined_inputs = joined_inputs.to(dtype=torch.float32)
        b, t, joined_dimension = joined_inputs.size()
        padded = torch.tensor(self.padding_vector, dtype=torch.float32,
                              device=joined_inputs.device).repeat(b, t, 1)

        if terminals is not None:
            terminal_mask = torch.clone(1 - terminals).repeat(1, 1, joined_inputs.shape[-1])
            joined_inputs = joined_inputs*terminal_mask+(1-terminal_mask)*padded

        state = joined_inputs[:,0,:self.observation_dim]
        ## [ B x T x embedding_dim ]
        # forward the GPT model
        reconstructed, aux_loss, feature = self.model(torch.cat([joined_inputs, terminals], dim=2), state)
        pred_trajectory = torch.reshape(reconstructed[:, :, :-1], shape=[b, t, joined_dimension])
        pred_terminals = reconstructed[:, :, -1, None]

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            #kl = torch.mean(-0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp(), dim=1), dim=0)
            weights = torch.cat([
                torch.ones(2, device=joined_inputs.device)*self.position_weight,
                torch.ones(self.observation_dim-2, device=joined_inputs.device),
                torch.ones(self.action_dim, device=joined_inputs.device) * self.action_weight,
                torch.ones(1, device=joined_inputs.device) * self.reward_weight,
                torch.ones(1, device=joined_inputs.device) * self.value_weight,
            ])
            mse = F.mse_loss(pred_trajectory, joined_inputs, reduction='none')*weights[None, None, :]

            first_action_loss = self.first_action_weight*F.mse_loss(joined_inputs[:, 0, self.observation_dim:self.observation_dim+self.action_dim],
                                                                    pred_trajectory[:, 0, self.observation_dim:self.observation_dim+self.action_dim])
            sum_reward_loss = self.sum_reward_weight*F.mse_loss(joined_inputs[:, :, -2].mean(dim=1),
                                                                pred_trajectory[:, :, -2].mean(dim=1))
            last_value_loss = self.last_value_weight*F.mse_loss(joined_inputs[:, -1, -1],
                                                                pred_trajectory[:, -1, -1])
            cross_entropy = F.binary_cross_entropy(pred_terminals, torch.clip(terminals.float(), 0.0, 1.0))
            reconstruction_loss = (mse*mask*terminal_mask).mean()+cross_entropy
            reconstruction_loss = reconstruction_loss + first_action_loss + sum_reward_loss + last_value_loss

            #reconstruction_loss = torch.sqrt((mse * mask).sum(dim=1)).mean()

            if self.model.ma_update:
                loss_vq = 0
            else:
                # loss_vq = F.mse_loss(latents, feature.detach())
                loss_vq = 0
            # Commitment objective
            # loss_commit = F.mse_loss(feature, latents.detach())
            loss_commit = aux_loss
            #loss_commit = 0
        else:
            reconstruction_loss = None
            loss_vq = None
            # Commitment objective
            loss_commit = None
        return reconstructed, reconstruction_loss, loss_vq, loss_commit


class TransformerPrior(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        # input embedding stem (+1 for stop token)
        self.tok_emb = nn.Embedding(config.K, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.state_emb = nn.Linear(config.observation_dim, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.K, bias=False)
        # self.head = EinLinear(config.transition_dim, config.n_embd, config.K, bias=False)
        self.observation_dim = config.observation_dim

        self.vocab_size = config.K
        self.block_size = config.block_size
        self.embedding_dim = config.n_embd
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, EinLinear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        if self.wt:
            decay.discard('head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, state, targets=None):
        """
            idx : [ B x T ]
            state: [ B ]
        """

        state = state.to(dtype=torch.float32)
        ## [ B x T x embedding_dim ]
        if not idx is None:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            token_embeddings = torch.cat([torch.zeros(size=(b, 1, self.embedding_dim)).to(token_embeddings), token_embeddings],
                                             dim=1)
        else:
            b = 1
            t = 0
            token_embeddings = torch.zeros(size=(b, 1, self.embedding_dim)).to(state)

        ## [ 1 x T+1 x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t+1, :] # each position maps to a (learnable) vector
        state_embeddings = self.state_emb(state)[:, None]
        ## [ B x T+1 x embedding_dim ]
        x = self.drop(token_embeddings + position_embeddings + state_embeddings)
        x = self.blocks(x)
        ## [ B x T+1 x embedding_dim ]
        x = self.ln_f(x)

        logits = self.head(x)
        logits = logits.reshape(b, t + 1, self.vocab_size)
        logits = logits[:,:t+1]

        # if we are given some desired targets also calculate the loss
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape([-1]), reduction='none')
            loss = loss.mean()
        else:
            loss = None

        return logits, loss


class LMHead(nn.Module):
    def __init__(self, n_embd, codebook_size, n_group):
        super().__init__()
        self.n_group = n_group
        self.net = nn.ModuleList([nn.Linear(n_embd, codebook_size) for _ in range(n_group)])
    def forward(self, x):
        out = [net(x) for net in  self.net]
        out = rearrange(out, 'g b t cl -> b t g cl')
        return out

class MultiGroupEmbedding(nn.Module):
    def __init__(self, n_embd, codebook_size, n_group):
        super().__init__()
        self.n_group = n_group
        self.embedding_layers = nn.ModuleList([nn.Embedding(codebook_size, n_embd) for _ in range(n_group)])
    def forward(self, idx):
        idx = rearrange(idx, 'b t g -> g b t')
        emb = [el(i) for i ,el in zip(idx, self.embedding_layers)] #g b t e
        emb = reduce(emb, 'g b t e -> b t e', 'sum')
        return emb

class MultiGroupMLP(nn.Module):
    def __init__(self, input_dim, output_dim, n_group):
        super().__init__()
        self.n_group = n_group
        self.mlp_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim))
            for _ in range(n_group)])
    def forward(self, x):
        x = [mlp(x) for mlp in self.mlp_layers]
        x = torch.stack(x, dim=0)
        x = rearrange(x, 'g b t out -> b t g out')
        return x
