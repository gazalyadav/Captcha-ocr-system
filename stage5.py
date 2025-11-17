# stage5.py — Advanced fine-tuning (Stage-5)
# - Uses previous checkpoint (PREV_CKPT) as starting weights
# - LR warmup + cosine annealing, weight decay, dropout in classifier
# - Optional SWA (Stochastic Weight Averaging)
# - Trains safely with MPS fallback enabled for CTC loss

import os
# Ensure MPS fallback for ctc if running on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time
import math
import random
from pathlib import Path
from typing import List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import timm

# try importing local utils (CaptchaDS + decode + CHARS) if present, otherwise minimal fallback
try:
    from utils import CaptchaDS, decode, CHARS
    HAVE_UTILS = True
except Exception:
    HAVE_UTILS = False
    import string
    CHARS = string.ascii_lowercase + string.ascii_uppercase + "0123456789"

# ---------------------- CONFIG ----------------------
DATA_DIR       = "./data/archive"            # training data (same as your earlier stages)
PREV_CKPT      = "finetuned_ocr.pth"         # weights to load (stage4 output)
OUT_CKPT       = "stage5_latest.pth"        # save final weights
SWA_CKPT       = "stage5_swa.pth"           # optional SWA snapshot
IMG_H, IMG_W   = 40, 360                    # wider width (you requested 300→360 possible)
BATCH_SIZE     = 16
EPOCHS         = 15
LR             = 5e-5                       # base LR (we use warmup then cosine)
WEIGHT_DECAY   = 1e-4                       # weight decay (L2)
DROPOUT_PROB   = 0.2                        # dropout in classifier
WARMUP_EPOCHS  = 2
SWA_START      = int(EPOCHS * 0.6)          # start SWA after 60% epochs
USE_SWA        = True                       # enable SWA (toggle)
SEED           = 42
NUM_WORKERS    = 0                           # change if you want parallel data loading

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}, PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")

# reproducibility
torch.manual_seed(SEED)
random.seed(SEED)

# ---------------------- HELPERS ----------------------
def get_files(folder: str) -> (List[str], List[str]):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    paths = [os.path.join(folder, f) for f in files]
    labels = [os.path.splitext(f)[0] for f in files]
    return paths, labels

# safe decode fallback (if utils.decode missing)
_idx2char = {i+1: c for i, c in enumerate(CHARS)}
_idx2char[0] = ""
def greedy_decode(preds: torch.Tensor) -> List[str]:
    """ Greedy CTC decoding from log-probabilities (batch x T x C) """
    arr = preds.argmax(2).cpu().numpy()
    results = []
    for seq in arr:
        s = ""
        prev = -1
        for p in seq:
            if p != prev and p != 0:
                s += _idx2char.get(p, "")
            prev = p
        results.append(s)
    return results

# ---------------------- MODEL ----------------------
class CRNN(nn.Module):
    def __init__(self, num_classes=len(CHARS) + 1, dropout=DROPOUT_PROB):
        super().__init__()
        # EfficientNet backbone (no classifier)
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.lstm = nn.LSTM(self.backbone.num_features, 256, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)  # 256*2
    def forward(self, x):
        f = self.backbone.forward_features(x)         # [B, C, H, W']
        f = f.mean(2).permute(0, 2, 1)                # [B, W, C] time-major as width
        x, _ = self.lstm(f)                           # [B, W, 512]
        x = self.dropout(x)
        return self.fc(x).log_softmax(2)              # log-probs for CTCLoss

# ---------------------- DATASET (fallback) ----------------------
if HAVE_UTILS:
    # use user's CaptchaDS (keeps same behavior)
    DatasetClass = CaptchaDS
else:
    from torch.utils.data import Dataset
    class CaptchaDS(Dataset):
        def __init__(self, paths, labels, aug=False):
            self.paths = paths
            self.labels = labels
            if aug:
                self.tf = T.Compose([
                    T.Resize((IMG_H, IMG_W)),
                    T.RandomApply([T.RandomAffine(degrees=5, shear=6, translate=(0.05,0.05))], p=0.35),
                    T.ToTensor(),
                ])
            else:
                self.tf = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor()])
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return self.tf(img), torch.tensor([ord(c) for c in self.labels[i]], dtype=torch.long)  # fallback (not ideal)

# collate function similar to previous code
def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), labels

# ---------------------- SETUP ----------------------
paths, labels = get_files(DATA_DIR)
print(f"Loaded {len(paths)} training samples from {DATA_DIR}")

train_ds = CaptchaDS(paths, labels, aug=True)   # heavy augmentation for fine-tuning
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, collate_fn=collate_fn)

model = CRNN().to(DEVICE)

# load previous weights if available
if os.path.exists(PREV_CKPT):
    print(f"Loading weights from {PREV_CKPT}")
    model.load_state_dict(torch.load(PREV_CKPT, map_location=DEVICE))
else:
    print("Warning: previous checkpoint not found. Training from scratch.")

# optimizer + scheduler
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# Cosine annealing scheduler with warmup handled manually
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS))

# optional SWA
if USE_SWA:
    swa_model = optim.swa_utils.AveragedModel(model)
    swa_start = SWA_START
    swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=LR*0.1)

ctc = nn.CTCLoss(blank=0, reduction="mean")   # standard CTCLoss

# ---------------------- TRAIN LOOP ----------------------
print("🔧 Stage-5 advanced fine-tuning started…\n")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    # LR warmup
    if epoch <= WARMUP_EPOCHS:
        warmup_mult = epoch / float(max(1, WARMUP_EPOCHS))
        for g in optimizer.param_groups:
            g["lr"] = LR * warmup_mult

    for imgs, lbls in train_loader:
        imgs = imgs.to(DEVICE)                 # images on device
        # flatten labels into 1D tensor expected by CTCLoss
        flat_lbls = torch.cat([torch.tensor([ord(c) for c in s], dtype=torch.long) if isinstance(s, str) else s for s in lbls])
        # NOTE: if using utils.CaptchaDS, lbls will be list of tensors already - keep compatibility
        if isinstance(flat_lbls, list):
            flat_lbls = torch.cat(flat_lbls)
        flat_lbls = flat_lbls.to(DEVICE)

        label_lens = torch.tensor([len(s) for s in lbls], dtype=torch.long)

        preds = model(imgs)   # [B, T, C] log-probs

        # prepare lengths
        pred_lens = torch.full((imgs.size(0),), preds.size(1), dtype=torch.long)

        # --- CTC may not be supported natively on MPS; run on CPU fallback when necessary ---
        # move tensors required by ctc to cpu (preserves graph when fallback enabled)
        try:
            # If MPS fallback is active, ctc will compute and autograd will work normally
            loss = ctc(preds.permute(1,0,2), flat_lbls, pred_lens, label_lens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
        except RuntimeError as e:
            # Fallback safety: compute CTC on CPU — PyTorch with fallback should still work if env var set.
            # If not, raise so user can set PYTORCH_ENABLE_MPS_FALLBACK=1 in environment.
            if "NotImplementedError" in str(e) or "not currently implemented for the MPS device" in str(e):
                raise RuntimeError(
                    "CTC op not available on MPS. Please set PYTORCH_ENABLE_MPS_FALLBACK=1 before running "
                    "or run on CPU. Example: export PYTORCH_ENABLE_MPS_FALLBACK=1"
                ) from e
            else:
                raise

        epoch_loss += batch_loss
        n_batches += 1

    # scheduler step (after epoch)
    if epoch <= WARMUP_EPOCHS:
        # keep warmup LR scaling
        pass
    else:
        if USE_SWA and epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

    avg_loss = epoch_loss / max(1, n_batches)
    lr_now = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch}/{EPOCHS}  |  LR {lr_now:.2e}  |  Loss {avg_loss:.4f}")

# finalize SWA if used
if USE_SWA:
    print("Updating batch-norm statistics for SWA model...")
    # build a temporary loader with small subset if very large dataset
    optim.swa_utils.update_bn(train_loader, swa_model)
    torch.save(swa_model.module.state_dict(), SWA_CKPT)
    print(f"Saved SWA checkpoint: {SWA_CKPT}")

# save final checkpoint
torch.save(model.state_dict(), OUT_CKPT)
total_time = time.time() - start_time
print(f"\n✅ Stage-5 completed. Saved: {OUT_CKPT}")
print(f"⏱️ Total time: {total_time/60:.2f} minutes")
