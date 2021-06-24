import torch
from torchvision.transforms import (
    Compose,
    Grayscale,    
    Normalize,
    Resize,
    RandomHorizontalFlip,
    ToTensor,    
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.data.masked_dataset import MaskedDataset

def pyramid_transform(mean=0, std=1):
    transform = {
        'head': Compose([
            RandomHorizontalFlip(),
            Grayscale(),
        ]),
        'large': Compose([
            Resize([128, 128]),
            ToTensor(),
            Normalize(mean=(mean), std=(std)),
        ]),
        'medium': Compose([
            Resize([64, 64]),
            ToTensor(),
            Normalize(mean=(mean), std=(std)),
        ]),
        'small': Compose([
            Resize([32, 32]),
            ToTensor(),
            Normalize(mean=(mean), std=(std)),
        ]),
    }
    def final_transform(img):
        flipped = transform['head'](img)
        return {
            'large': transform['large'](flipped),
            # 'medium': transform['medium'](flipped),
            # 'small': transform['small'](flipped),
        }
    return final_transform

def calculate_mean_std(ds):
    ''' Don't use normalize on transform'''
    old_transform = ds.transform
    ds.transform = pyramid_transform()

    loader = DataLoader(
        ds,
        batch_size=10,
        num_workers=1,
        shuffle=False
    )

    mean = 0.
    std = 0.
    nb_samples = 0.
    for (batch, _) in loader:
        data = batch['large']
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    ds.transform = old_transform
    return mean, std

def loader_generator(loader):
    current = 0
    iterator = iter(loader)
    while True:
        if current >= len(loader):
            current = 0
            iterator = iter(loader)
        yield next(iterator)
        current += 1    

class Provider:
    
    def __init__(self, config):
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory

        self.rec_bs = config.reconstruction_batch_size
        self.cnt_bs = config.contrastive_batch_size

        self.adv_image_root = config.adversarial_image_root
        self.adv_bs = config.adversarial_batch_size
        self.adv_mean = config.adversarial_image_mean
        self.adv_std = config.adversarial_image_std
        self.adv_real_label = config.adversarial_real_label
        self.adv_fake_label = config.adversarial_fake_label
        
        adv_transform = pyramid_transform(self.adv_mean, self.adv_std)
        adv_ds = ImageFolder(self.adv_image_root, transform=adv_transform)
        adv_loader = DataLoader(adv_ds, batch_size=self.adv_bs, shuffle=True)
        self.adv_gen = loader_generator(adv_loader)

        rec_ds = MaskedDataset(config)
        rec_loader = DataLoader(rec_ds, shuffle=True, 
            batch_size=self.rec_bs, num_workers=self.num_workers,
            pin_memory=self.pin_memory)
        self.rec_gen = loader_generator(rec_loader)
    
    def adversarial(self, labels=False, device=None):
        batch, _ = next(self.adv_gen)
        bs = batch['large'].size(0)
        if labels:
            batch['label_real'] = torch.full((bs,), self.adv_real_label)
            batch['label_fake'] = torch.full((bs,), self.adv_real_label)
        if device is not None:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
        return batch, (self.adv_mean, self.adv_std)
    
    def reconstruction(self, device=None):
        batch = next(self.rec_gen)
        if device is not None:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
        return batch        
    
    def contrastive():
        pass        