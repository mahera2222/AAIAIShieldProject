import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.efficientnet_lite0 import EfficientNetLite0Tamper
import os

# --------------------------
# Device Setup
# --------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", DEVICE)

# --------------------------
# Data Transforms
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------
# Dataset
# --------------------------
train_data = datasets.ImageFolder("data/train", transform=transform)
val_data   = datasets.ImageFolder("data/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=16)

# --------------------------
# Model
# --------------------------
model = EfficientNetLite0Tamper(num_classes=2).to(DEVICE)

# Weighted loss to fix dataset imbalance
class_weights = torch.tensor([1.0, 2.0]).to(DEVICE)  
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# --------------------------
# Training Parameters
# --------------------------
EPOCHS = 15   

history = {"loss": [], "acc": []}

# --------------------------
# Training Loop
# --------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
	history["loss"].append(avg_loss)

    # Validation Accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            _, predicted = torch.max(out, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total

    # Save training history
    history["loss"].append(epoch_loss)
    history["acc"].append(acc)

    scheduler.step(acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss={epoch_loss:.4f} | Val Acc={acc:.4f}")

# --------------------------
# Save model + history
# --------------------------
os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

torch.save(model.state_dict(), "saved_models/efficient_lite0.pth")
torch.save(history, "results/history.pth")

print("\nModel saved → saved_models/efficient_lite0.pth")
print("Training history saved → results/history.pth\n")
