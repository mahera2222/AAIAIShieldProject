import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class EfficientNetTamper(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Load pretrained EfficientNet-B2
        self.base = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

        # Replace final classifier
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)
