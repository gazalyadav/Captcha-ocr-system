"""
EVALUATION SCRIPT FOR CAPTCHA OCR MODEL
---------------------------------------
This script loads the trained CRNN model and evaluates its performance on a test dataset.

Metrics reported:
- Full CAPTCHA accuracy (exact string match)
- Character Error Rate (CER)
"""

# =====================================
# IMPORTS AND LIBRARY EXPLANATIONS
# =====================================

import os                                    # for reading file paths from folder
import torch                                  # for model loading, tensor ops
import torch.nn as nn                         # neural network layers
from torch.utils.data import DataLoader, Dataset  # data loading utilities
import torchvision.transforms as T            # image preprocessing transforms
from PIL import Image                         # loading images
import string                                 # for character set (A-Z, a-z, 0-9)
import timm                                   # pre-trained EfficientNet backbone (Transfer Learning)
from difflib import SequenceMatcher           # used for CER calculation

# =====================================
# CONFIGURATION
# =====================================

DATA_DIR = "data/archive"                     # path to test CAPTCHA dataset
MODEL_PATH = "final_ocr.pth"                  # trained model weights
IMG_H, IMG_W = 40, 300                        # standard input resolution used during training
BATCH_SIZE = 32

# Character vocabulary used in CAPTCHA dataset
CHARS = string.ascii_lowercase + string.ascii_uppercase + "0123456789"

# Mappings for label encoding/decoding
idx2char = {i+1: c for i, c in enumerate(CHARS)}
idx2char[0] = ""                              # CTC blank token

# Select device: Apple MPS (Mac GPU) or CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

# =====================================
# CER FUNCTION
# =====================================

def cer(ref, hyp):
    """
    Character Error Rate = 1 - similarity ratio
    Lower CER = better performance
    """
    return 1 - SequenceMatcher(None, ref, hyp).ratio()

# =====================================
# DECODER FOR CTC OUTPUT
# =====================================

def decode(preds):
    """
    Convert CTC time-step predictions to text using greedy decoding.
    Remove repeated characters and blanks (0).
    """
    preds = preds.argmax(2).cpu().numpy()     # take highest-probability class at each timestep
    results = []

    for seq in preds:
        s = ""
        last = -1
        for p in seq:
            if p != last and p != 0:         # collapse duplicates + ignore blank token
                s += idx2char[p]
            last = p
        results.append(s)
    return results

# =====================================
# DATASET CLASS
# =====================================

class CAPTCHADS(Dataset):
    """Dataset loader for test CAPTCHA images."""

    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
        self.tf = T.Compose([
            T.Resize((IMG_H, IMG_W)),        # scale image to training resolution
            T.ToTensor()                     # convert to tensor [0,1]
        ])

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]

    def __len__(self):
        return len(self.paths)

def collate_fn(batch):
    """Stack batch images and keep label list intact."""
    imgs, labels = zip(*batch)
    return torch.stack(imgs), labels

# =====================================
# MODEL DEFINITION (CRNN)
# CNN + BiLSTM + CTC
# =====================================

class CRNN(nn.Module):
    """
    CRNN Architecture:
    - EfficientNet-B0 CNN backbone (feature extractor)
    - BiLSTM layers (sequence modeling)
    - Linear layer (character classification)
    - LogSoftmax for CTC loss compatibility
    """

    def __init__(self, nc=len(CHARS)+1):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.lstm = nn.LSTM(self.backbone.num_features, 256, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, nc)

    def forward(self, x):
        f = self.backbone.forward_features(x) # CNN features [B,C,H,W]
        f = f.mean(2).permute(0,2,1)          # Global avg pool H ⇒ [B,W,C]
        x,_ = self.lstm(f)                    # sequence learning
        return self.fc(x).log_softmax(2)      # CTC-ready output

# =====================================
# LOAD TEST DATA
# =====================================

files  = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg",".png"))]
paths  = [os.path.join(DATA_DIR, f) for f in files]
labels = [os.path.splitext(f)[0] for f in files]

ds = CAPTCHADS(paths, labels)
loader = DataLoader(ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# =====================================
# LOAD MODEL WEIGHTS
# =====================================

model = CRNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =====================================
# EVALUATION LOOP
# =====================================

correct = 0
total   = 0
cer_sum = 0

with torch.no_grad():
    for imgs, gts in loader:
        imgs = imgs.to(device)
        preds = decode(model(imgs))

        for p, g in zip(preds, gts):
            if p == g:
                correct += 1
            cer_sum += cer(g, p)
            total += 1

acc = correct/total
avg_cer = cer_sum/total

print(f"✅ Final Captcha Accuracy: {acc*100:.2f}%")
print(f"✅ CER: {avg_cer:.4f}")
