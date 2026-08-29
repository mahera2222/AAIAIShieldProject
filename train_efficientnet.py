import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.efficientnet_lite0 import EfficientNetLite0Tamper


# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Dataset
train_data = datasets.ImageFolder(
    "data/train",
    transform=transform
)

val_data = datasets.ImageFolder(
    "data/val",
    transform=transform
)

print("Classes:", train_data.classes)
print("Class mapping:", train_data.class_to_idx)
print("Training samples:", len(train_data))
print("Validation samples:", len(val_data))


# Data loaders
train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=16,
    shuffle=False
)


# Model
model = EfficientNetLite0Tamper(
    num_classes=2
).to(DEVICE)


# Calculate class weights from the training dataset
targets = torch.tensor(train_data.targets)
class_counts = torch.bincount(targets)

class_weights = (
    len(targets)
    / (len(class_counts) * class_counts.float())
).to(DEVICE)

print("Class counts:", class_counts.tolist())
print("Class weights:", class_weights.cpu().tolist())


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# Training parameters
EPOCHS = 15

history = {
    "loss": [],
    "acc": []
}

best_val_acc = 0.0

os.makedirs(
    "saved_models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# Training loop
for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0

    for imgs, labels in train_loader:

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = (
        epoch_loss
        / len(train_loader)
    )


    # Validation
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for imgs, labels in val_loader:

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)

            predicted = outputs.argmax(
                dim=1
            )

            correct += (
                predicted == labels
            ).sum().item()

            total += labels.size(0)


    val_acc = correct / total

    history["loss"].append(
        avg_loss
    )

    history["acc"].append(
        val_acc
    )

    scheduler.step(
        val_acc
    )


    # Save best model
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "saved_models/efficient_lite0.pth"
        )

        print(
            f"Best model updated "
            f"(Val Acc={best_val_acc:.4f})"
        )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss={avg_loss:.4f} | "
        f"Val Acc={val_acc:.4f}"
    )


# Save training history
torch.save(
    history,
    "results/history.pth"
)

print()
print(
    "Best validation accuracy:",
    f"{best_val_acc:.4f}"
)
print(
    "Best model saved → "
    "saved_models/efficient_lite0.pth"
)
print(
    "Training history saved → "
    "results/history.pth"
)
