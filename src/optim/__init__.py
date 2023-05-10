from optim.autoencoder import Autoencoder
from optim.finetune import Finetuner
from optim.simclr import SimCLR

def get_methods(name: str):

    available_methods = {
        'Autoencoder': Autoencoder,
        'Finetune': Finetuner,
        'SimCLR': SimCLR
    }

    return available_methods[name]