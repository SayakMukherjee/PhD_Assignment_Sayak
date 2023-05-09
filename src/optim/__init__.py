from optim.autoencoder import Autoencoder
from optim.finetune import Finetuner

def get_methods(name: str):

    available_methods = {
        'Autoencoder': Autoencoder,
        'Finetune': Finetuner
    }

    return available_methods[name]