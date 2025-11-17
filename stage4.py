"""
----------------------------- FINE-TUNING MODULE -----------------------------

This script performs the final fine-tuning stage of the CAPTCHA OCR model.
The objective is to further improve accuracy and robustness by training
the already-trained CRNN model using:

• A lower learning rate (1e-5)  
• Wider input width (360 px)  
• Strong augmentations for improved generalization  

The output of this script is an enhanced model checkpoint: `finetuned_ocr.pth`.

-------------------------------------------------------------------------------
"""

# ----------------------------- IMPORTS --------------------------------------
import os
import torch
import timm
import string
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import CaptchaDS, decode, CHARS   # custom utilities for dataset, decoding

# --------------------------- CONFIGURATION -----------------------------------
DATA_DIR = "./data/archive"      # training dataset folder
CKPT = "final_ocr.pth"           # previously trained model checkpoint
IMG_W = 360                      # wider width for better sequence capture
BATCH = 16                       # batch size
EPOCHS = 10                      # number of fine-tuning epochs
LR = 1e-5                        # very low LR for subtle weight updates

# Select appropriate device (Apple MPS GPU or CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"


# ----------------------------- MODEL DEFINITION ------------------------------
class CRNN(nn.Module):
    """
    CRNN Model:
    - EfficientNet-B0 backbone for feature extraction (CNN)
    - Bidirectional LSTM for sequence modeling
    - Fully connected layer for per-timestep character classification
    - Log-softmax output for CTC Loss compatibility
    """
    def __init__(self, nc=len(CHARS) + 1):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.lstm = nn.LSTM(
            self.backbone.num_features, 
            256, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(512, nc)  # 256 * 2 for BiLSTM output

    def forward(self, x):
        # Extract convolutional features
        f = self.backbone.forward_features(x)

        # Convert to sequence features: [Batch, Width, Channels]
        f = f.mean(2).permute(0, 2, 1)

        # Process through LSTM for sequence learning
        x, _ = self.lstm(f)

        # Output class logits (log-probabilities)
        return self.fc(x).log_softmax(2)


# ----------------------------- DATA LOADING ----------------------------------
# Collect all image paths and labels from the dataset folder
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
paths = [os.path.join(DATA_DIR, f) for f in files]
labels = [os.path.splitext(f)[0] for f in files]  # filename = ground truth label

# Load dataset with augmentations enabled
train_dataset = CaptchaDS(paths, labels, aug=True)

# DataLoader with batch collation logic
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH,
    shuffle=True,
    collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b])
)


# ----------------------------- MODEL SETUP -----------------------------------
model = CRNN().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))  # Load previous weights

# Define optimizer and CTC Loss
opt = optim.Adam(model.parameters(), lr=LR)
ctc = nn.CTCLoss(blank=0)


# ----------------------------- TRAINING LOOP ---------------------------------
print("🔧 Fine-tuning started...\n")

for epoch in range(1, EPOCHS + 1):

    model.train()
    total_loss = 0

    for imgs, lbls in train_loader:

        imgs = imgs.to(device)

        # Flatten labels for CTC loss
        flat_lbls = torch.cat(lbls).to(device)

        # Compute label lengths per sample
        label_lengths = torch.tensor([len(l) for l in lbls])

        # Forward pass
        preds = model(imgs)

        # Prediction sequence length (uniform for all images)
        pred_lengths = torch.full((imgs.size(0),), preds.size(1), dtype=torch.long)

        # Compute CTC loss
        loss = ctc(preds.permute(1, 0, 2), flat_lbls, pred_lengths, label_lengths)

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f}")


# ----------------------------- SAVE FINAL MODEL ------------------------------
torch.save(model.state_dict(), "finetuned_ocr.pth")
print("\n✅ Fine-tuning completed successfully!")
print("✅ Saved model as: finetuned_ocr.pth")
