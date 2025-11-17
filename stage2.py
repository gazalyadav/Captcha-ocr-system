"""
stage2.py  (Warm-Up / Transfer-Learning Stage)
---------------------------------------------
This script performs the first stage of training for the CAPTCHA OCR model.

Training Method:
----------------
1. Load EfficientNet-B0 backbone (ImageNet pretrained)
2. Freeze backbone for first N epochs (transfer learning warm-up)
3. Train only BiLSTM + classification head initially
4. Gradually unfreeze CNN backbone for full fine-tuning
5. Save checkpoints every epoch

Purpose:
--------
Stabilize training and prevent early catastrophic forgetting before full fine-tuning.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import string
import timm

#########################################
# Configurations
#########################################
IMG_H, IMG_W = 40, 250            # CAPTCHA resized resolution
BATCH_SIZE = 16                   # fits Mac MPS GPU memory
LR = 5e-5                         # warm-up learning rate
EPOCHS = 30
FREEZE_EPOCHS = 10                # epochs to freeze backbone
AUG_START_EPOCH = 5               # begin augmentation after warm-up
DATA_DIR = "/Users/gazalyadav_/Desktop/captcha/data/archive"

# Character vocabulary (62-class CAPTCHA)
CHARS = string.ascii_lowercase + string.ascii_uppercase + "0123456789"
char2idx = {c: i+1 for i, c in enumerate(CHARS)}
idx2char = {i+1: c for i, c in enumerate(CHARS)}
idx2char[0] = ""   # CTC blank index

#########################################
# Decode CTC predictions
#########################################
def decode(preds):
    preds = preds.argmax(2).cpu().numpy()
    texts = []
    for seq in preds:
        s = ""
        prev = -1
        for p in seq:
            if p != prev and p != 0:
                s += idx2char[p]
            prev = p
        texts.append(s)
    return texts

#########################################
# Load dataset file paths
#########################################
def get_files_labels(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".png")]
    paths = [os.path.join(folder, f) for f in files]
    labels = [os.path.splitext(f)[0] for f in files]
    return paths, labels

#########################################
# Dataset class
#########################################
class CaptchaDS(Dataset):
    def __init__(self, paths, labels, aug=False):
        if aug:
            self.t = T.Compose([
                T.Resize((IMG_H, IMG_W)),
                T.ToTensor(),
                T.RandomApply([T.GaussianBlur(3)], p=0.1),
                T.RandomApply([T.RandomAffine(degrees=3, shear=5, translate=(0.03,0.03))], p=0.2)
            ])
        else:
            self.t = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor()])

        self.p = paths
        self.l = labels

    def __getitem__(self, idx):
        img = Image.open(self.p[idx]).convert("RGB")
        img = self.t(img)
        label = torch.tensor([char2idx[c] for c in self.l[idx]], dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.p)

def collate(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), labels

#########################################
# Model: EfficientNet-B0 + BiLSTM + CTC
#########################################
class CRNN(nn.Module):
    def __init__(self, num_classes=len(CHARS)+1):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.lstm = nn.LSTM(self.backbone.num_features, 256, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        f = f.mean(dim=2).permute(0, 2, 1)  # [batch, width, channels]
        x, _ = self.lstm(f)
        return self.fc(x).log_softmax(2)

#########################################
# Training Loop
#########################################
if __name__ == "__main__":

    # load dataset paths
    paths, labels = get_files_labels(DATA_DIR)

    # SELECT 20k sample warm-up subset
    paths, labels = paths[:20000], labels[:20000]

    # split train/val
    split = int(0.9 * len(paths))
    train_ds = CaptchaDS(paths[:split], labels[:split], aug=False)
    val_ds = CaptchaDS(paths[split:], labels[split:], aug=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Training on:", device)

    model = CRNN().to(device)
    torch.set_float32_matmul_precision('high')

    # freeze backbone initially (transfer-learning phase)
    for p in model.backbone.parameters():
        p.requires_grad = False

    opt = optim.Adam(model.parameters(), lr=LR)
    ctc = nn.CTCLoss(blank=0)

    for epoch in range(1, EPOCHS+1):

        # enable augmentation after warm-up
        if epoch == AUG_START_EPOCH:
            train_ds = CaptchaDS(paths[:split], labels[:split], aug=True)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

        # unfreeze CNN backbone after N epochs
        if epoch == FREEZE_EPOCHS:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print("Backbone unfrozen (full fine-tuning)")

        model.train()
        total_loss = 0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            flat_lbls = torch.cat(lbls).to(device)
            lbl_lens = torch.tensor([len(l) for l in lbls], dtype=torch.long)
            preds = model(imgs)
            pred_lens = torch.full((imgs.size(0),), preds.size(1), dtype=torch.long)

            loss = ctc(preds.permute(1,0,2), flat_lbls, pred_lens, lbl_lens)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # quick validation logging
        model.eval()
        vimg, vlbl = next(iter(val_loader))
        vtxt = decode(model(vimg.to(device)))[0]

        print(f"Epoch {epoch}/{EPOCHS} Loss={avg_loss:.4f} Pred={vtxt} GT={vlbl[0]}")

        torch.save(model.state_dict(), f"stage2_checkpoint_epoch{epoch}.pth")

print("✅ Warm-up Stage Training Finished")
