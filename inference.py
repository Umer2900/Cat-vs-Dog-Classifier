# inference.py
import torch
from torchvision import transforms
from PIL import Image
from models.model_factory import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load("models/saved/best_model.pth", map_location=device)
MODEL_NAME = checkpoint["model_name"] 

# Build and load model
model = build_model(MODEL_NAME, num_classes=2, pretrained=False)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()

# Transform based on model
if MODEL_NAME == "lenet5":
    size = 32
    norm = ((0.5,)*3, (0.5,)*3)
else:
    size = 224
    norm = ((0.485,0.456,0.406), (0.229,0.224,0.225))

transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])

def predict(image):
    """Predict class and probability"""
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][class_idx].item()
    return class_idx, confidence
