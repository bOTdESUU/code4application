from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MultioutputWrapper
from torchmetrics.classification.accuracy import Accuracy
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat, pack, unpack
import inspect

from pretrain.models.autoencoders.nets.helper import symlog, symexp
from pretrain.models.autoencoders.vae import BaseVAE
from lfq_bet.models.nets.gpt import GPT, MLP


#TODO: 1. divide vocab into subspace, done
#2. multi token prediction

class LFQBeT(LightningModule):

    def __init__(
        self,
        act_tok:LightningModule,
        state_tok:LightningModule,
        gpt_cfg,
        compile: bool,
        n_input_group: int = 1,
        n_act_group: int =1 ,
        n_pred_act: int = 1,
        gamma: float = 2.0,
        **kwargs
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.act_tok = act_tok 
        self.state_tok = state_tok
        #TODO: implement method to get info from tokenizer
        self.input_emb = MultiGroupEmbedding(n_embd=gpt_cfg.n_embd, 
                                            n_group=2,
                                            codebook_size=256)
        self.gpt = GPT(gpt_cfg)
        self.act_head = MultiGroupMLP(n_embd=gpt_cfg.n_embd,
                                    codebook_size=256,
                                    n_group=2)
        self.state_head = LMHead(n_embd=gpt_cfg.n_embd,
                                codebook_size=256,
                                n_group=2)
        for embd, head in zip(self.input_emb.embedding_layers, self.state_head.net):
            embd.weight = head.weight
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying 

        # loss function
        # self.criterion = FocalLoss(gamma=gamma)
        self.gpt_token_loss = nn.CrossEntropyLoss()
        self.rec_loss = nn.MSELoss()
        self.setup_metrics()


    def setup_metrics(self):
        self.metrics = nn.ModuleDict({
            "train/total_loss": MeanMetric(),
            "train/action_token_loss": MeanMetric(),
            "train/action_token_acc": Accuracy(task='multiclass',num_classes=256),
            "train/action_rec_loss": MeanMetric(),
            "train/state_token_loss": MeanMetric(),
            "train/state_token_acc": Accuracy(task='multiclass',num_classes=256),
            "train/state_rec_loss": MeanMetric(),
            "val/total_loss": MeanMetric(),
            "val/action_token_loss": MeanMetric(),
            "val/action_token_acc": Accuracy(task='multiclass',num_classes=256),
            "val/action_rec_loss": MeanMetric(),
            "val/state_token_loss": MeanMetric(),
            "val/state_token_acc": Accuracy(task='multiclass',num_classes=256),
            "val/state_rec_loss": MeanMetric(),
        })
        # loss_dict = {
        #         "classification_loss": cbet_loss.detach().cpu().item(),
        #         "offset_loss": offset_loss.detach().cpu().item(),
        #         "total_loss": loss.detach().cpu().item(),
        #         "equal_total_code_rate": equal_total_code_rate,
        #         "equal_single_code_rate": equal_single_code_rate,
        #         "equal_single_code_rate2": equal_single_code_rate2,
        #         "action_diff": action_diff.detach().cpu().item(),
        #         "action_diff_tot": action_diff_tot.detach().cpu().item(),
        #         "action_diff_mean_res1": action_diff_mean_res1.detach().cpu().item(),
        #         "action_diff_mean_res2": action_diff_mean_res2.detach().cpu().item(),
        #         "action_diff_max": action_diff_max.detach().cpu().item(),
        #     }

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # obs, act, goal = batch
        act_token = self.act_tok.tokenize(batch['act'])
        state_token = self.state_tok.tokenize(batch['obs']) #b t g
        state_emb = self.input_emb(state_token)
        gpt_out = self.gpt(state_emb)
        act_logit = self.act_head(gpt_out)
        if act_logit.dim() == 4:
            action_token_loss = [self.gpt_token_loss(l, tok) for l, tok in 
                    zip(rearrange(act_logit, 'b t g e -> g (b t) e'),
                        rearrange(act_token, 'b t g -> g (b t)'))]
            action_token_loss = torch.stack(action_token_loss, dim=0).mean(dim=0)
            # loss = rearrange(loss, 'g b t -> b t g')
        else:
            action_token_loss = self.gpt_token_loss(act_logit, act_token)
        
        pred_act_token = torch.argmax(act_logit, dim=-1).detach()
        pred_act = self.act_tok.detokenize(pred_act_token)
        act_rec_loss = self.rec_loss(pred_act, batch['act'])
        
        state_logit = self.state_head(gpt_out)
        if state_logit.dim() == 4:
            state_token_loss = [self.gpt_token_loss(l, tok) for l, tok in 
                    zip(rearrange(state_logit, 'b t g e -> g (b t) e'),
                        rearrange(state_token, 'b t g -> g (b t)'))]
            state_token_loss = torch.stack(state_token_loss, dim=0).mean(dim=0)
        else:
            state_token_loss = self.gpt_token_loss(state_logit, state_token)
        
        pred_state_token = torch.argmax(state_logit, dim=-1)
        pred_state = self.state_tok.detokenize(pred_state_token)
        state_rec_loss = self.rec_loss(pred_state, batch['obs'])

        #TODO: implement weigted loss
        total_loss = action_token_loss + 1e-1 * state_token_loss + 1e-3*(act_rec_loss + state_rec_loss)  
        
        lossbreakdown = {
            "action_token_loss": action_token_loss,
            "state_token_loss": state_token_loss,
            "action_rec_loss": act_rec_loss,
            "state_rec_loss": state_rec_loss,
            "total_loss": total_loss
        }
        return total_loss, lossbreakdown

    def _log_metrics(self, loss_breakdown: Dict, prefix: str) -> None:

        self.metrics[f"{prefix}/total_loss"](loss_breakdown.pop("total_loss"))
        self.log(f"{prefix}/total_loss", self.metrics[f"{prefix}/total_loss"], prog_bar=True)
        for loss_name, loss_value in loss_breakdown.items():
            loss_name = f"{prefix}/{loss_name}"
            self.metrics[loss_name](loss_value)
            self.log(loss_name, self.metrics[loss_name], on_step=False, on_epoch=True, prog_bar=False)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # loss, preds, targets = self.model_step(batch)
        total_loss, loss_breakdown = self.model_step(batch)
        self._log_metrics(loss_breakdown, "train")
        # return loss or backpropagation will fail
        return total_loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        total_loss, loss_breakdown = self.model_step(batch)
        self._log_metrics(loss_breakdown, "val")
        return total_loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:

        pass

    def eval_on_vqbet_env(self,
                          obs: torch.Tensor,
                          goal: Optional[torch.Tensor], 
                          act: Optional[torch.Tensor],
                          ):
        #TODO: sequence length
        obs = self._pad_seq(obs, 10)
        obs = self.state_tok.tokenize(obs).clone().detach()
        state_emb = self.input_emb(obs)
        gpt_out = self.gpt(state_emb)
        logit = self.act_head(gpt_out)
        pred_act_token = torch.argmax(logit, dim=-1)
        pred_act = self.act_tok.detokenize(pred_act_token)
        return pred_act, None, None

    @staticmethod
    def _pad_seq(seq, length):
        return F.pad(seq, (0, 0, length - seq.size(1), 0), mode='replicate')
    
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.act_tok = torch.compile(self.act_tok)
            self.state_tok = torch.compile(self.state_tok)
            self.input_emb = torch.compile(self.input_emb)
            self.gpt = torch.compile(self.gpt)
            self.act_head = torch.compile(self.act_head)
            self.state_head = torch.compile(self.state_head)


    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        
        # configure gpt optimizer
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.gpt.named_parameters()}
        param_dict.update({pn: p for pn, p in self.act_head.named_parameters()})
        param_dict.update({pn: p for pn, p in self.input_emb.named_parameters()})
        
        if self.hparams.act_tok_lr is not None:
            param_dict.update({pn: p for pn, p in self.act_tok.named_parameters()})
        else:
            self.act_tok.eval().freeze()
        if self.hparams.state_tok_lr is not None:
            param_dict.update({pn: p for pn, p in self.state_tok.named_parameters()})
        else:
            self.state_tok.eval().freeze()

        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': 2e-4},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # self.log.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # self.log.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        optimizer = self.hparams.optimizer(optim_groups,)
        # self.log.info(f"using fused AdamW: {use_fused}")
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return({
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            })

        # if self.hparams.act_tok_optimizer is None:
        #     self.act_tok.eval().freeze()
        # else:
        #     act_tok_opt = self.hparams.act_tok_optimizer(self.act_tok.parameters())
        #     opt.append({"optimizer": act_tok_opt})
        # if self.hparams.state_tok_optimizer is None:
        #     self.state_tok.eval().freeze()
        # else:
        #     opt.append({"optimizer": self.hparams.state_tok_optimizer(self.state_tok.parameters())})
        # return opt


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target_index):
        # logit = rearrange(logit,'b t ... -> (b t) ...')
        # target_index = rearrange(target_index, 'b t ... -> (b t) ...')
        logpt = F.log_softmax(logit, dim=-1)
        logpt = logpt.gather(1, target_index.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

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
    def __init__(self, n_embd, codebook_size, n_group):
        super().__init__()
        self.n_group = n_group
        self.mlp_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, codebook_size))
            for _ in range(n_group)])
    def forward(self, x):
        x = [mlp(x) for mlp in self.mlp_layers]
        x = torch.stack(x, dim=0)
        x = rearrange(x, 'g b t cl -> b t g cl')
        return x

