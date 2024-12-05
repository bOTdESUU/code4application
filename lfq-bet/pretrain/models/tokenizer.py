import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torch import nn

from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torch import nn

from pretrain.models.autoencoders.nets.helper import symlog, symexp
from pretrain.models.autoencoders.vae import BaseVAE

class BaseTokenizer(LightningModule):
    def __init__(
        self,
        vae: BaseVAE,
        use_symlog: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        key = None #key use to accese the batch
    ) -> None:

        super().__init__()

        self.save_hyperparameters(logger=False)
        
        self.use_symlog = use_symlog
        self.vae = vae
        self.key = key
        
        if self.use_symlog:
            self.input_proj = symlog
            self.output_proj = symexp
        else:
            self.input_proj =nn.Identity()
            self.output_proj = nn.Identity()

        self.recon_loss = torch.nn.MSELoss()

        # # for tracking best so far validation accuracy
        self.setup_metrics()

    def tokenize(self, x) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.vae.encode(x)
        return x
    
    def detokenize(self, idx):
        x = self.vae.decode(idx)
        return self.output_proj(x)

    def setup_metrics(self):
        self.metrics = nn.ModuleDict( {
            "train/rec_loss": MeanMetric(),
            "train/total_loss": MeanMetric(),
            'train/per_sample_entropy': MeanMetric(),
            "train/batch_entropy": MeanMetric(),
            "train/commitment": MeanMetric(),
            "val/rec_loss": MeanMetric(),
            "val/total_loss": MeanMetric(),
            'val/per_sample_entropy': MeanMetric(),
            "val/batch_entropy": MeanMetric(),
            "val/commitment": MeanMetric(),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        x = self.input_proj(x)
        x = self.vae(x)
        x = self.output_proj(x)
        return x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch[self.key]
        x = self.input_proj(x)
        tar = x.clone().detach()
        x, index, aux_loss, loss_breakdown = self.vae.model_step(x)
        rec_loss = self.recon_loss(x, tar)
        total_loss = rec_loss + aux_loss

        loss_breakdown = loss_breakdown._asdict()
        loss_breakdown["rec_loss"] = rec_loss
        loss_breakdown["total_loss"] = total_loss

        return x, index, total_loss, loss_breakdown
    
    def _log_metrics(self, loss_breakdown: Dict[str, MeanMetric], prefix: str) -> None:
        for loss_name, loss_value in loss_breakdown.items():
            loss_name = f"{prefix}/{loss_name}"
            self.metrics[loss_name](loss_value)
            self.log(loss_name, self.metrics[loss_name], on_step=False, on_epoch=True, prog_bar=True)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # loss, preds, targets = self.model_step(batch)
        rec_seq, _, total_loss, loss_breakdown = self.model_step(batch)
        self._log_metrics(loss_breakdown, "train")
        # return loss or backpropagation will fail
        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        rec_seq, _, total_loss, loss_breakdown = self.model_step(batch)
        self._log_metrics(loss_breakdown, "val")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.vae = torch.compile(self.vae)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}



