import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.efficientnet_lite0 import EfficientNetLite0Tamper
from sklearn.metrics import classification_report, roc_curve, auc


# ---------------------------------------------------
# 🔧 SETTINGS
# ---------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "saved_models/efficient_lite0.pth"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


# ---------------------------------------------------
# 📌 Load model
# ---------------------------------------------------
def load_model():
    model = EfficientNetLite0Tamper(num_classes=2).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model



# ---------------------------------------------------
# 📌 Prepare data for visualization
# ---------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_data = datasets.ImageFolder("data/val", transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)


# ---------------------------------------------------
# 📌 SAMPLE PREDICTION GRID
# ---------------------------------------------------
def save_sample_predictions(model):
    X, y = next(iter(test_loader))
    X = X.to(DEVICE)
    preds = model(X).argmax(dim=1).cpu()

    fig, axs = plt.subplots(3, 3, figsize=(7, 7))
    idx = 0

    for r in range(3):
        for c in range(3):
            img = X[idx].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize for display

            axs[r, c].imshow(img)
            axs[r, c].set_title(f"P={preds[idx]} | T={y[idx]}")
            axs[r, c].axis("off")
            idx += 1

    plt.tight_layout()
    path = f"{RESULT_DIR}/sample_predictions.png"
    plt.savefig(path)
    plt.close()
    print("Saved sample predictions:", path)


# ---------------------------------------------------
# 📌 LOSS CURVE
# ---------------------------------------------------
def plot_loss_curve(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Training Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.legend()

    path = f"{RESULT_DIR}/loss_curve.png"
    plt.savefig(path)
    plt.close()
    print("Saved loss curve:", path)


# ---------------------------------------------------
# 📌 ACCURACY CURVE
# ---------------------------------------------------
def plot_accuracy_curve(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["acc"], label="Accuracy", color="green", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.grid(True)
    plt.legend()

    path = f"{RESULT_DIR}/accuracy_curve.png"
    plt.savefig(path)
    plt.close()
    print("Saved accuracy curve:", path)


# ---------------------------------------------------
# 📌 COMBINED LOSS + ACCURACY CURVE
# ---------------------------------------------------
def plot_loss_accuracy(history):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="red")
    ax1.plot(history["loss"], color="red", linewidth=2, label="Loss")
    ax1.tick_params(axis='y', labelcolor="red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="blue")
    ax2.plot(history["acc"], color="blue", linewidth=2, label="Accuracy")
    ax2.tick_params(axis='y', labelcolor="blue")

    plt.title("Loss + Accuracy Curve")
    plt.grid(True)

    path = f"{RESULT_DIR}/loss_accuracy_curve.png"
    plt.savefig(path)
    plt.close()
    print("Saved combined curve:", path)


# ---------------------------------------------------
# 📌 ROC CURVE
# ---------------------------------------------------
def plot_roc_curve(model, loader, device, save_path="results/roc_curve.png"):

    model.eval()
    y_true = []
    y_prob = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)

            probs = torch.softmax(logits, dim=1)[:, 1]  # class 1 -> tampered

            y_true.extend(y.numpy())
            y_prob.extend(probs.cpu().numpy())

    # Compute ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr, tpr,
        color="darkorange",
        lw=2,
        label=f"ROC Curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Tampered Detection")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()
    print("Saved ROC curve:", save_path)


# ---------------------------------------------------
# 📌 MAIN
# ---------------------------------------------------
if __name__ == "__main__":

    print("📌 Loading model...")
    model = load_model()

    print("📌 Generating sample predictions...")
    save_sample_predictions(model)

    print("📌 Loading training history...")
    history = torch.load("results/history.pth")

    print("📌 Plotting curves...")
    plot_loss_curve(history)
    plot_accuracy_curve(history)
    plot_loss_accuracy(history)

    print("📌 Plotting ROC Curve...")
    plot_roc_curve(model, test_loader, DEVICE)

    print("\n🎉 Visualization completed! Check the results/ folder.\n")
