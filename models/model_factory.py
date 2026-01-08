import torch.nn as nn
from torchvision import models
from .lenet5 import LeNet5

def build_model(name, num_classes=2, pretrained=True):
    name = name.lower()

    if name == "lenet5":
        model = LeNet5(num_classes)

    elif name == "alexnet":
        model = models.alexnet( weights=models.AlexNet_Weights.DEFAULT if pretrained else None )
        model.classifier[6] = nn.Linear( model.classifier[6].in_features, num_classes )

    elif name == "resnet18":
        model = models.resnet18( weights=models.ResNet18_Weights.DEFAULT if pretrained else None )
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError("Unknown model name")

    return model
