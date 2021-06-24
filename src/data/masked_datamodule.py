from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.data.masked_dataset import MaskedDataset


class MaskedDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.reconstruction_batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        
    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None): # Stage {'fit'|'test'}
        self.ds = MaskedDataset(self.config) 
        
    def train_dataloader(self):
        return DataLoader(self.ds, shuffle=True, batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=self.pin_memory)