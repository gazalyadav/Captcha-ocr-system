"""
---------------- AUTOMATIC CAPTCHA OCR PREDICTION ----------------

This script automatically loads the most recent captcha image
from the downloads folder and predicts its text using a trained CRNN model.
"""

# --------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------
import os
import torch
import timm
import string
from PIL import Image
from pathlib import Path
import torchvision.transforms as T

# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------

# Path to folder containing downloaded captcha images
CAPTCHA_DIR = Path("/Users/gazalyadav_/Desktop/captcha/downloads")

# Path to trained model weights
MODEL_PATH = "stage5_final.pth"

# Input image dimensions (used during training)
IMG_H, IMG_W = 40, 300

# Character set: a-z, A-Z, 0-9
CHARS = string.ascii_lowercase + string.ascii_uppercase + "0123456789"

# Mapping for CTC decoding
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
idx2char[0] = ""

# Select device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# --------------------------------------------------------------
# CTC GREEDY DECODER
# --------------------------------------------------------------
def decode(preds):
    preds = preds.argmax(2).cpu().numpy()
    texts = []
    for seq in preds:
        s = ""
        last = -1
        for p in seq:
            if p != last and p != 0:
                s += idx2char[p]
            last = p
        texts.append(s)
    return texts[0]

# --------------------------------------------------------------
# CRNN MODEL ARCHITECTURE
# --------------------------------------------------------------
class CRNN(torch.nn.Module):
    def __init__(self, nc=len(CHARS) + 1):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.lstm = torch.nn.LSTM(
            self.backbone.num_features, 256, 2, batch_first=True, bidirectional=True
        )
        self.fc = torch.nn.Linear(512, nc)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        f = f.mean(2).permute(0, 2, 1)
        x, _ = self.lstm(f)
        return self.fc(x)

# --------------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------------
model = CRNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --------------------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------------------
transform = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor()
])

# --------------------------------------------------------------
# FUNCTION: Get latest captcha image
# --------------------------------------------------------------
def get_latest_captcha():
    if not CAPTCHA_DIR.exists():
        raise FileNotFoundError(f"Captcha directory not found: {CAPTCHA_DIR}")
    images = list(CAPTCHA_DIR.glob("*.png")) + list(CAPTCHA_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No captcha images found in {CAPTCHA_DIR}")
    latest = max(images, key=lambda p: p.stat().st_mtime)
    return latest

# --------------------------------------------------------------
# FUNCTION: Predict image text
# --------------------------------------------------------------
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img).log_softmax(2)
        text = decode(out)
    return text

# --------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------
# --------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------
if __name__ == "__main__":
    try:
        latest_img = get_latest_captcha()
        print(f"🖼️  Using latest captcha: {latest_img.name}")
        result = predict_image(latest_img)
        print(f"✅ Predicted Text: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")