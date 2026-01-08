import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),     # (B, 3, 32, 32) → (B, 6, 28, 28)
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                 # → (B, 6, 14, 14)

            nn.Conv2d(6, 16, kernel_size=5),    # → (B, 16, 10, 10)
            nn.Tanh(),
            nn.AvgPool2d(2, 2)                  # → (B, 16, 5, 5)
        )

        # Classifier 
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.classifier(x)
        return x
