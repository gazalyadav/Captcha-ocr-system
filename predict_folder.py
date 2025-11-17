"""
predict_all.py
Predicts text from all CAPTCHA images in a given folder using a trained CRNN model.
Requirements:
    pip install torch torchvision timm pillow
"""

import os
import torch
import timm
import string
from PIL import Image
import torchvision.transforms as T

# ---------------- CONFIG ----------------
MODEL_PATH = "stage5_final.pth"
IMAGE_DIR  = "/Users/gazalyadav_/Desktop/captcha/downloads"
IMG_H, IMG_W = 40, 300

CHARS = string.ascii_lowercase + string.ascii_uppercase + "0123456789"
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
idx2char[0] = ""   # CTC blank

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️ Using device: {device.upper()}")

# ---------------- MODEL ----------------
class CRNN(torch.nn.Module):
    def __init__(self, nc=len(CHARS) + 1):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.lstm = torch.nn.LSTM(
            self.backbone.num_features, 256, 2,
            batch_first=True, bidirectional=True
        )
        self.fc = torch.nn.Linear(512, nc)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        f = f.mean(2).permute(0, 2, 1)  # (B, C, W) → (B, W, C)
        x, _ = self.lstm(f)
        return self.fc(x).log_softmax(2)

# ---------------- LOAD MODEL ----------------
model = CRNN().to(device)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"✅ Loaded model: {MODEL_PATH}")

# ---------------- TRANSFORM ----------------
transform = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor()
])

# ---------------- DECODE FUNCTION ----------------
def decode(preds):
    preds = preds.argmax(2).cpu().numpy()[0]
    s, last = "", -1
    for p in preds:
        if p != 0 and p != last:
            s += idx2char[p]
        last = p
    return s

# ---------------- PREDICT FUNCTION ----------------
def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
    return decode(logits)

# ---------------- MAIN EXECUTION ----------------
if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(f"❌ Image folder not found: {IMAGE_DIR}")

files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(("png", "jpg", "jpeg"))])
if not files:
    print("❌ No CAPTCHA images found in folder.")
    exit()

print(f"\n🔍 Found {len(files)} images in: {IMAGE_DIR}")
print("--------------------------------------------------")

for f in files:
    img_path = os.path.join(IMAGE_DIR, f)
    try:
        pred = predict(img_path)
        print(f"🧩 {f} → {pred}")
    except Exception as e:
        print(f"⚠️ Error processing {f}: {e}")

print("--------------------------------------------------")
print(f"✅ Completed predictions for {len(files)} images.")
print("📁 Folder:", IMAGE_DIR)
print("🏁 DONE.")
