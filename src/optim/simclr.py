#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-May-2023
#
# ---------------------------------------------------------------------------
# Pytorch Lightning trainer class for SimCLR
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from models import SimCLRLoss, ContractiveLoss

class SimCLR(pl.LightningModule):

    def __init__(self, config, model):
        super().__init__()

        self.config = config
        self.model = model

        self.loss_fn = SimCLRLoss()
        self.regularizer = self.configure_regularizer()

        self.train_loss = torchmetrics.MeanMetric()

    def configure_regularizer(self):
        regularizer = None

        if self.config.train.use_reg:
            regularizer = ContractiveLoss()

        return regularizer

    def configure_optimizers(self):
        if self.config.optimize.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.optimize.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.config.optimizer}")
        return optimizer

    def training_step(self, batch, batch_idx):

        indexes, images = batch[0], batch[1]

        # Considering 2 views
        images = torch.cat([images[0], images[1]], dim=0)
        indexes = indexes.repeat(2) 

        if self.regularizer is not None:

            images.requires_grad_(True)
            images.retain_grad()

            embeddings = self.model(images)
            con_loss = self.regularizer(embeddings, images)
            loss = self.loss_fn(embeddings, indexes)  + (1e-4 * con_loss)

            images.requires_grad_(False)

        else:
            
            embeddings = self.model(images)
            loss = self.loss_fn(embeddings, indexes) 

        self.train_loss.update(loss)

        # Log step-level loss
        self.log("train/loss_step", loss)
        return loss

    def on_train_epoch_end(self):
        # Log the epoch-level training accuracy
        self.log('train/loss', self.train_loss.compute())
        self.train_loss.reset()