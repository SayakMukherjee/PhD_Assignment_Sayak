#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-May-2023
#
# ---------------------------------------------------------------------------
# Contains the ResNet backbone and classifier used in SimCLR
# ---------------------------------------------------------------------------

import torchvision
import torch.nn as nn

from models.basenet import BaseNet
from collections import OrderedDict


class MLP(nn.Module):

    def __init__(self, embedding_size, num_classes):
        super(MLP, self).__init__()

        self.net = nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('fcout', nn.Linear(embedding_size, num_classes))
            ]))

    def forward(self, x, apply_log_softmax = False):

        x = self.net(x)

        return x
    
class ResNet(BaseNet):

    def __init__(self, config):
        super(ResNet, self).__init__()

        self.config = config

        self.rep_dim = 256

        self.model = torchvision.models.resnet18()
        self.model.fc = nn.Sequential(OrderedDict([
            ('fcin', nn.Linear(512, self.rep_dim)),
        ]))

        self.classifier = MLP(self.rep_dim, 10)

    def forward(self, x):

        embeddings = self.model(x[0])         
        outputs = self.classifier(embeddings)

        return outputs
    
class ResNetBackbone(BaseNet):

    def __init__(self, config):
        super(ResNetBackbone, self).__init__()

        self.config = config

        self.rep_dim = 256
        self.proj_hidden_dim = 2048
        self.proj_output_dim = 256

        self.model = torchvision.models.resnet18()
        self.model.fc = nn.Sequential(OrderedDict([
            ('fcin', nn.Linear(512, self.rep_dim)),
        ]))

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.rep_dim, self.proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
        )

    def forward(self, x):

        embeddings = self.model(x)

        outputs = self.projector(embeddings)         

        return outputs