from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def cifar10_loaders(
    data_dir: str | Path,
    batch_size: int,
    workers: int = 2,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(data_dir, train=True, transform=transform, download=download)
    test_set = datasets.CIFAR10(data_dir, train=False, transform=transform, download=download)
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common)
    return train_loader, test_loader
