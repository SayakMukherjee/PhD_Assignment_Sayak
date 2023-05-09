#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-May-2023
#
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

class Finetuner(pl.LightningModule):

    def __init__(self, config, model):
        super().__init__()

        self.config = config
        self.model = model

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_loss = torchmetrics.MeanMetric()
        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def configure_optimizers(self):
        if self.config.optimize.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.optimize.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.config.optimizer}")
        return optimizer

    def _step(self, x, y):

        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch, batch_idx):

        _, inputs, targets = batch[0], batch[1], batch[2]

        loss, y_hat = self._step(inputs, targets)
        preds = torch.argmax(y_hat, dim=1)

        self.train_loss.update(loss)
        self.train_acc(preds, targets)

        # Log step-level loss and accuracy
        self.log("train/loss_step", loss)
        self.log("train/acc_step", self.train_acc)
        return loss

    def test_step(self, batch, batch_idx):
        _, inputs, targets = batch[0], batch[1], batch[2]

        loss, y_hat = self._step(inputs, targets)
        preds = torch.argmax(y_hat, dim=1)

        self.test_acc(preds, targets)

        # Log step-level loss and accuracy
        self.log("test/loss", loss)
        self.log("test/acc_step", self.test_acc)
        return loss

    def on_train_epoch_end(self):
        # Log the epoch-level training accuracy
        self.log('train/loss', self.train_loss.compute())
        self.train_loss.reset()