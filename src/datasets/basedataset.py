#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-May-2023
#
# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

from abc import abstractmethod

class BaseDataset():

    def __init__(self, config):
        self.config = config

        self.pretrain_dataset = None
        self.train_dataset = None
        self.test_dataset = None

        self.pretrain_loader = None
        self.train_loader = None
        self.test_loader = None

    def get_pretrain_loader(self):
        return self.pretrain_loader

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader
    
    @abstractmethod
    def init_pretrain_dataset(self):
        pass

    @abstractmethod
    def init_train_dataset(self):
        pass

    @abstractmethod
    def init_test_dataset(self):
        pass