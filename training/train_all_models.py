import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.model_factory import build_model
from training.train_utils import train_one_epoch
from training.evaluate import evaluate

MODEL_NAMES = ["lenet5", "alexnet", "resnet18"]       # vgg take a lot of time
NUM_EPOCHS = 2

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc = 0.0
best_model_name = None
best_model_state = None

for MODEL_NAME in MODEL_NAMES:
    print(f"\nTraining {MODEL_NAME.upper()}")

    if MODEL_NAME == "lenet5":
        size = 32
        norm = ((0.5,)*3, (0.5,)*3)
    else:
        size = 224
        norm = ((0.485,0.456,0.406), (0.229,0.224,0.225))

    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, train_transform)
    test_ds = datasets.ImageFolder(TEST_DIR, test_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = build_model(MODEL_NAME, num_classes=2, pretrained=(MODEL_NAME!="lenet5")).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 1e-3 if MODEL_NAME == "lenet5" else 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(f"Epoch {epoch+1}: Test Acc = {test_acc:.4f}")

    if test_acc > best_acc:
        best_acc = test_acc
        best_model_name = MODEL_NAME
        best_model_state = model.state_dict()


os.makedirs("models/saved", exist_ok=True)  # create folder if not exist

torch.save({
    "model_name": best_model_name,
    "state_dict": best_model_state
}, "models/saved/best_model.pth")

print(f"\nBest Model: {best_model_name} | Accuracy: {best_acc:.4f}")
