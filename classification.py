import torch
import torch.nn as nn

from models.vgg11 import VGG11
from models.layers import CustomDropout

# Input size fixed per VGG paper (224x224)
_FLAT_DIM = 512 * 7 * 7


class VGG11Classifier(nn.Module):
    """VGG11 encoder + classification head.

    Args:
        num_classes: Number of output classes (default: 37 for Oxford-IIIT Pets).
        in_channels: Number of input image channels (default: 3).
        dropout_p:   Dropout probability for CustomDropout (default: 0.5).
    """

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        self.encoder = VGG11(in_channels=in_channels)

        self.classifier = nn.Sequential(
            nn.Linear(_FLAT_DIM, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            logits: [B, num_classes]
        """
        x = self.encoder(x)          # [B, 512, 7, 7]
        x = torch.flatten(x, 1)      # [B, 512*7*7]
        x = self.classifier(x)       # [B, num_classes]
        return x
