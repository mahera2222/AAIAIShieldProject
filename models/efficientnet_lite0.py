import torch
import torch.nn as nn
import timm

class EfficientNetLite0Tamper(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # EfficientNet-Lite0 backbone used for training and inference
        self.base = timm.create_model(
            "efficientnet_lite0",
            pretrained=True,        # Initialize with ImageNet pretrained weights
            num_classes=0           # remove default classifier
        )

        # Binary classification head: clean vs. tampered
        self.classifier = nn.Linear(self.base.num_features, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x
