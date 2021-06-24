from torchvision.transforms import (
    Compose,
    Grayscale,    
    Normalize,
    Resize,
    RandomHorizontalFlip,
    ToTensor,    
)
from torch.utils.data import DataLoader

mean = 0.1834
std = 0.2670





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
            'medium': transform['medium'](flipped),
            'small': transform['small'](flipped),
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