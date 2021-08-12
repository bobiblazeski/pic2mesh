# pyright: reportMissingImports=false
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.data.surface_dataset import SurfaceDataset


class SurfaceDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.fast_batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.train_ds = SurfaceDataset(self.config)         
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=self.pin_memory)

    