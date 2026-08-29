import torch.nn as nn
import timm


class EfficientNetLite0Tamper(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # EfficientNet-Lite0 backbone
        self.base = timm.create_model(
            "efficientnet_lite0",
            pretrained=pretrained,
            num_classes=0
        )

        # Binary classification head
        self.classifier = nn.Linear(
            self.base.num_features,
            num_classes
        )

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x
