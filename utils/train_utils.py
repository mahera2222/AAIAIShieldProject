import torch
import torch.nn as nn
from tqdm import tqdm
from utils.focal_loss import FocalLoss
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def train_detector(model, train_loader, test_loader, device, epochs=12):

    criterion = FocalLoss(alpha=0.6, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = {"loss": [], "acc": []}

    model.train()

    for epoch in range(1, epochs + 1):
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        correct = 0
        total = 0
        running_loss = 0

        for X, y in loop:
            X, y = X.to(device), y.to(device)

            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = preds.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

        acc = correct / total
        history["loss"].append(running_loss)
        history["acc"].append(acc)
        scheduler.step(acc)

        print(f"Epoch {epoch}: Loss={running_loss:.4f} | Accuracy={acc:.4f}")

    return history


def evaluate_model(model, test_loader, device):

    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            _, predicted = preds.max(1)
            all_preds.extend(predicted.cpu())
            all_true.extend(y.cpu())

    report = classification_report(all_true, all_preds)
    print("\nClassification Report:\n")
    print(report)

    with open("results/report.txt", "w") as f:
        f.write(report)
