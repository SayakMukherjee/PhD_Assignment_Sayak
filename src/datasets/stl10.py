#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-May-2023
#
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

from datasets.basedataset import BaseDataset
from datasets.utils import SimCLRCollate, SaltPepperNoise
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets, transforms
from typing import Callable, Optional
from pathlib import Path


class DatasetWrapper(tv_datasets.STL10):

    def __init__(self,
                 root: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        super(DatasetWrapper, self).__init__(root=root,
                                             split=split,
                                             transform=transform,
                                             target_transform=target_transform,
                                             download=download)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        return (index, *data)

class STL10(BaseDataset):

    def __init__(self, config):

        super(STL10, self).__init__(config)

        self.config = config

        self.collate_fn = None
        self.transform = None

        if config.dataset.use_collate:
            print(f"Using collate function")
            self.collate_fn = SimCLRCollate(input_size=32,
                                            min_scale=0.2,
                                            gaussian_blur=0.0)

            if config.dataset.is_corrupted:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    SaltPepperNoise()
                    ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor()
                    ])
        else:
            print(f"Using transform")

            if config.dataset.is_corrupted:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

            else:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    SaltPepperNoise()])

        self.init_pretrain_dataset()
        self.init_train_dataset()
        self.init_test_dataset()

    def init_pretrain_dataset(self):

        print(f"Loading STL10 unlabelled data")

        if self.transform is None:
            self.pretrain_dataset = DatasetWrapper(root = Path(self.config.dataset.root),
                                                split = 'unlabeled',
                                                download = True)
        else:
            print('with transform')
            self.pretrain_dataset = DatasetWrapper(root = Path(self.config.dataset.root),
                                                split = 'unlabeled',
                                                download = True,
                                                transform = self.transform)

        if self.collate_fn is None:
            print('without collate')
            self.pretrain_loader = DataLoader(self.pretrain_dataset,
                                           batch_size=self.config.train.batch_size,
                                           num_workers=self.config.dataset.num_workers,
                                           drop_last=True)
        else:
            self.pretrain_loader = DataLoader(self.pretrain_dataset,
                                           batch_size=self.config.train.batch_size,
                                           collate_fn=self.collate_fn,
                                           num_workers=self.config.dataset.num_workers,
                                           drop_last=True)

    def init_train_dataset(self):

        print(f"Loading STL10 train data")

        if self.transform is None:
            self.train_dataset = DatasetWrapper(root = Path(self.config.dataset.root),
                                                download = True)
        else:
            self.train_dataset = DatasetWrapper(root = Path(self.config.dataset.root),
                                                download = True,
                                                transform = self.transform)

        if self.collate_fn is None:
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.config.train.batch_size,
                                           num_workers=self.config.dataset.num_workers,
                                           drop_last=True)
        else:
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.config.train.batch_size,
                                           collate_fn=self.collate_fn,
                                           num_workers=self.config.dataset.num_workers,
                                           drop_last=True)

    def init_test_dataset(self):

        print(f"Loading STL10 test data")

        if self.transform is None:
            self.test_dataset = DatasetWrapper(root = Path(self.config.dataset.root),
                                               split = 'test',
                                               download=True)
        else:
            self.test_dataset = DatasetWrapper(root = Path(self.config.dataset.root),
                                               split = 'test',
                                               download=True,
                                               transform=self.transform)

        if self.collate_fn is None:
            self.test_loader = DataLoader(self.test_dataset,
                                          batch_size=self.config.train.batch_size,
                                          num_workers=self.config.dataset.num_workers)
        else:
            self.test_loader = DataLoader(self.test_dataset,
                                          batch_size=self.config.train.batch_size,
                                          collate_fn=self.collate_fn,
                                          num_workers=self.config.dataset.num_workers)