#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-May-2023
#
# ---------------------------------------------------------------------------
#  Pytorch Lightning trainer class for autoencoder
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from models import PSELoss, ContractiveLoss

class Autoencoder(pl.LightningModule):

    def __init__(self, config, model):
        super().__init__()

        self.config = config
        self.model = model

        self.loss_fn = self.configure_loss()
        self.regularizer = self.configure_regularizer()

        self.train_loss = torchmetrics.MeanMetric()

    def configure_regularizer(self):
        regularizer = None

        if self.config.train.use_reg:
            regularizer = ContractiveLoss()

        return regularizer

    def configure_loss(self):
        if self.config.train.loss == 'MSE':
            loss = nn.MSELoss()

        elif self.config.train.loss == 'PSE':
            # TODO: Update this
            kernel_size = [5,5]
            sigma = [11,11]

            loss = PSELoss(kernel_size=kernel_size, sigma=sigma)
        else:
            raise NotImplementedError(f"Loss {self.config.train.loss}")

        return loss

    def configure_optimizers(self):
        if self.config.optimize.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.optimize.lr,
                                         weight_decay=self.config.optimize.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.config.optimizer}")
        return optimizer

    def _step(self, x):

        if self.regularizer is not None:

            x.requires_grad_(True)
            x.retain_grad()

            recons, hidden_rep = self.model(x)
            con_loss = self.regularizer(hidden_rep, x)
            loss = self.loss_fn(recons, x) + (1e-4 * con_loss)

            x.requires_grad_(False)

        else:
            recons, hidden_rep = self.model(x)
            loss = self.loss_fn(recons, x)
        return loss

    def training_step(self, batch, batch_idx):

        _, inputs, _ = batch[0], batch[1], batch[2]

        loss = self._step(inputs)
        self.train_loss.update(loss)

        # Log step-level loss
        self.log("train/loss_step", loss)
        return loss

    def test_step(self, batch, batch_idx):
        _, inputs, _ = batch[0], batch[1], batch[2]

        loss = self._step(inputs)

        # Log test loss
        self.log("test/loss", loss)
        return loss

    def on_train_epoch_end(self):
        # Log the epoch-level training accuracy
        self.log('train/loss', self.train_loss.compute())
        self.train_loss.reset()