import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler


def get_dataloaders(batch_size=32):
    """
    Create training and validation dataloaders.

    The training loader uses a WeightedRandomSampler to reduce
    the effect of class imbalance.
    """

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.RandomRotation(5),
        transforms.ToTensor()
    ])

    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_path = "data/train"
    val_path = "data/val"

    train_ds = datasets.ImageFolder(
        train_path,
        transform=transform_train
    )

    val_ds = datasets.ImageFolder(
        val_path,
        transform=transform_val
    )

    # Handle class imbalance in the training set
    targets = [
        label
        for _, label in train_ds.samples
    ]

    class_counts = torch.bincount(
        torch.tensor(targets)
    )

    class_weights = 1.0 / class_counts.float()

    sample_weights = torch.tensor([
        class_weights[label]
        for label in targets
    ])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, train_ds.classes
