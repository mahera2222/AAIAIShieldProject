import os
import sys

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from models.efficientnet_lite0 import EfficientNetLite0Tamper


# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


MODEL_PATH = "saved_models/efficient_lite0.pth"
RESULT_DIR = "results"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# Preprocessing must match training/inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation dataset
val_data = datasets.ImageFolder(
    "data/val",
    transform=transform
)

val_loader = DataLoader(
    val_data,
    batch_size=16,
    shuffle=False
)


def load_model():
    model = EfficientNetLite0Tamper(
        num_classes=2,
        pretrained=False
    ).to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


def save_sample_predictions(model):
    images, labels = next(
        iter(val_loader)
    )

    images = images.to(DEVICE)

    with torch.no_grad():
        predictions = model(
            images
        ).argmax(dim=1).cpu()

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7, 7)
    )

    index = 0

    for row in range(3):
        for column in range(3):

            image = (
                images[index]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )

            image = (
                image - image.min()
            ) / (
                image.max()
                - image.min()
                + 1e-9
            )

            axes[row, column].imshow(
                image
            )

            axes[row, column].set_title(
                f"P={predictions[index]} "
                f"| T={labels[index]}"
            )

            axes[row, column].axis(
                "off"
            )

            index += 1

    plt.tight_layout()

    path = os.path.join(
        RESULT_DIR,
        "sample_predictions.png"
    )

    plt.savefig(
        path
    )

    plt.close()

    print(
        "Saved sample predictions:",
        path
    )


def plot_loss_curve(history):
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["loss"],
        label="Training Loss",
        linewidth=2
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training Loss Curve"
    )

    plt.grid(
        True
    )

    plt.legend()

    path = os.path.join(
        RESULT_DIR,
        "loss_curve.png"
    )

    plt.savefig(
        path
    )

    plt.close()

    print(
        "Saved loss curve:",
        path
    )


def plot_accuracy_curve(history):
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["acc"],
        label="Validation Accuracy",
        linewidth=2
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Validation Accuracy Curve"
    )

    plt.grid(
        True
    )

    plt.legend()

    path = os.path.join(
        RESULT_DIR,
        "accuracy_curve.png"
    )

    plt.savefig(
        path
    )

    plt.close()

    print(
        "Saved accuracy curve:",
        path
    )


def plot_loss_accuracy(history):
    fig, ax1 = plt.subplots(
        figsize=(8, 5)
    )

    ax1.set_xlabel(
        "Epoch"
    )

    ax1.set_ylabel(
        "Loss"
    )

    ax1.plot(
        history["loss"],
        linewidth=2,
        label="Loss"
    )

    ax2 = ax1.twinx()

    ax2.set_ylabel(
        "Accuracy"
    )

    ax2.plot(
        history["acc"],
        linewidth=2,
        label="Accuracy"
    )

    plt.title(
        "Loss and Accuracy"
    )

    path = os.path.join(
        RESULT_DIR,
        "loss_accuracy_curve.png"
    )

    plt.savefig(
        path
    )

    plt.close()

    print(
        "Saved combined curve:",
        path
    )


def evaluate_model(model):
    y_true = []
    y_pred = []
    y_prob = []

    model.eval()

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = outputs.argmax(
                dim=1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

            y_prob.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    report = classification_report(
        y_true,
        y_pred,
        target_names=val_data.classes
    )

    print()
    print("Classification Report:")
    print(report)

    report_path = os.path.join(
        RESULT_DIR,
        "report.txt"
    )

    with open(
        report_path,
        "w"
    ) as file:
        file.write(
            report
        )

    print(
        "Saved classification report:",
        report_path
    )

    return y_true, y_pred, y_prob


def plot_confusion_matrix(
    y_true,
    y_pred
):
    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        matrix
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        range(len(val_data.classes)),
        val_data.classes
    )

    plt.yticks(
        range(len(val_data.classes)),
        val_data.classes
    )

    for i in range(
        matrix.shape[0]
    ):
        for j in range(
            matrix.shape[1]
        ):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )

    path = os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    )

    plt.tight_layout()

    plt.savefig(
        path
    )

    plt.close()

    print(
        "Saved confusion matrix:",
        path
    )


def plot_roc_curve(
    y_true,
    y_prob
):
    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC Curve (AUC = {roc_auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - Tampered Detection"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        True
    )

    path = os.path.join(
        RESULT_DIR,
        "roc_curve.png"
    )

    plt.savefig(
        path
    )

    plt.close()

    print(
        "Saved ROC curve:",
        path
    )


if __name__ == "__main__":

    print(
        "Loading EfficientNet-Lite0 model..."
    )

    model = load_model()

    print(
        "Generating sample predictions..."
    )

    save_sample_predictions(
        model
    )

    print(
        "Loading training history..."
    )

    history = torch.load(
        "results/history.pth",
        map_location="cpu"
    )

    print(
        "Generating training curves..."
    )

    plot_loss_curve(
        history
    )

    plot_accuracy_curve(
        history
    )

    plot_loss_accuracy(
        history
    )

    print(
        "Evaluating validation set..."
    )

    y_true, y_pred, y_prob = evaluate_model(
        model
    )

    plot_confusion_matrix(
        y_true,
        y_pred
    )

    plot_roc_curve(
        y_true,
        y_prob
    )

    print()
    print(
        "Evaluation complete. "
        "Check the results/ folder."
    )
