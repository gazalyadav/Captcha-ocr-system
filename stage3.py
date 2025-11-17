"""
stage3.py
Final fine-tuning stage of the CAPTCHA OCR pipeline.

Purpose:
--------
This stage performs final end-to-end fine-tuning of the CRNN model
(EfficientNet-B0 backbone + BiLSTM + CTC loss) using the full dataset,
after transfer learning from Stage 2.

Key Features:
-------------
- Loads Stage-2 checkpoint
- Enables augmentations for robustness
- Uses Adam optimizer + CTCLoss
- Saves best performing model as final_ocr.pth
"""

import os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import timm
from math import inf
from utils import CaptchaDS, decode, cer, CHARS   # common utilities

############################################
# Configurations
############################################
DATA_DIR = "./data/archive"
BATCH_SIZE = 16
EPOCHS = 20
LR = 3e-5   # small LR for fine-tuning
STAGE2_CKPT = "stage2_checkpoint_epoch15.pth"

############################################
# Load image file paths and labels
############################################
def get_files(path):
    files = [f for f in os.listdir(path) if f.endswith(".jpg")]
    paths = [os.path.join(path, f) for f in files]
    labels = [f.split(".")[0] for f in files]      # filename = label
    return paths, labels


############################################
# CRNN Model (EfficientNet-B0 + BiLSTM + CTC)
############################################
class CRNN(nn.Module):
    def __init__(self, num_classes=len(CHARS)+1):
        super().__init__()
        # EfficientNet-B0 as CNN feature extractor
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)

        # Bi-directional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            self.backbone.num_features, 256,
            num_layers=2, batch_first=True, bidirectional=True
        )

        # Linear classifier for each timestep (sequence output)
        self.fc = nn.Linear(512, num_classes)  # 256*2

    def forward(self, x):
        # CNN feature map
        f = self.backbone.forward_features(x)

        # collapse height dimension → [BATCH, WIDTH, CHANNELS]
        f = f.mean(2).permute(0, 2, 1)

        # sequence prediction
        x, _ = self.lstm(f)

        return self.fc(x).log_softmax(2)  # CTC expects log-probs


############################################
# Stage-3 Training Function
############################################
def train_stage3():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    paths, labels = get_files(DATA_DIR)
    split = int(0.95 * len(paths))  # 95% train, 5% validation

    # Dataset + light augmentation for robustness
    train_ds = CaptchaDS(paths[:split], labels[:split], aug=True)
    val_ds   = CaptchaDS(paths[split:], labels[split:], aug=False)

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b])
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b])
    )

    # Load Stage-2 model
    model = CRNN().to(device)
    model.load_state_dict(torch.load(STAGE2_CKPT, map_location=device))

    opt = optim.Adam(model.parameters(), lr=LR)
    ctc = nn.CTCLoss(blank=0)
    best_loss = inf

    print(f"✅ Stage-3 fine-tuning on {device}")

    ########################################
    # Training Loop
    ########################################
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device)

            # flatten label lists into 1D tensor
            flat_labels = torch.cat(lbls).to(device)

            # compute label lengths per sample
            label_lens = torch.tensor([len(l) for l in lbls]).long()

            preds = model(imgs)
            pred_lens = torch.full((imgs.size(0),), preds.size(1), dtype=torch.long)

            loss = ctc(preds.permute(1, 0, 2), flat_labels, pred_lens, label_lens)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Quick validation preview
        vimgs, vlbls = next(iter(val_loader))
        preds = decode(model(vimgs.to(device)))[0]
        gt = "".join([CHARS[i-1] for i in vlbls[0]])

        print(f"Epoch {epoch}/{EPOCHS} | Loss {avg_loss:.4f} | Pred: {preds} | GT: {gt}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "final_ocr.pth")
            print("✔ Best model saved")

    print("🎯 Stage-3 Training Completed")


############################################
# Run
############################################
if __name__ == "__main__":
    train_stage3()
