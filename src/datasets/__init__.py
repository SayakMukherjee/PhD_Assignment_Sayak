from datasets.basedataset import BaseDataset
from datasets.stl10 import STL10

def get_dataset(name: str):
    
    available_datasets = {
        'STL10': STL10,
    }

    return available_datasets[name]