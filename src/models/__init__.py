from models.basenet import BaseNet
from models.lenet_ae import LeNet_Autoencoder
from models.lenet import LeNet
from models.losses import SimCLRLoss, PSELoss, ContractiveLoss
from models.resnet import ResNet, ResNetBackbone

def get_model(name: str):
    
    available_models = {
        'LeNet': LeNet,
        'LeNet_Autoencoder': LeNet_Autoencoder,
        'ResNet': ResNet,
        'ResNetBackbone': ResNetBackbone
    }

    return available_models[name]