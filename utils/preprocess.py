import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os

def get_dataloaders(batch_size=32):

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_path = "data/train"
    test_path = "data/test"

    train_ds = datasets.ImageFolder(train_path, transform=transform_train)
    test_ds = datasets.ImageFolder(test_path, transform=transform_test)

    # --- FIX CLASS IMBALANCE ---
    targets = [label for _, label in train_ds]
    class_counts = torch.bincount(torch.tensor(targets))
    class_weights = 1.0 / class_counts.float()
    sample_weights = torch.tensor([class_weights[t] for t in targets])

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_ds.classes
