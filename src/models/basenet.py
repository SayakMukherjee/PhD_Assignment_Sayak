#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-May-2023
# 
# ---------------------------------------------------------------------------
# Contains the base dataset class
# ---------------------------------------------------------------------------

import torch.nn as nn
import numpy as np


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        """Forward pass logic

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def summary(self):
        """Summary of the model
        """
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)