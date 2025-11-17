"""
utils.py
Utility functions and dataset class used for CAPTCHA OCR training and evaluation.
Includes:
- Character vocabulary mapping
- Image preprocessing
- CTC-friendly decoding
- Character Error Rate (CER) calculation
- PyTorch Dataset class with optional augmentation
"""

import string, torch, torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

#############################################
# Character Dictionary (0 = CTC blank token)
#############################################

# All possible CAPTCHA characters: a-z, A-Z, 0-9
CHARS = string.ascii_lowercase + string.ascii_uppercase + "0123456789"

# Map characters to numeric IDs (1–62). 0 reserved for blank token in CTC.
char2idx = {c: i+1 for i, c in enumerate(CHARS)}

# Reverse mapping: ID → character
idx2char = {i+1: c for i, c in enumerate(CHARS)}
idx2char[0] = ""   # blank for CTC decoding

#############################################
# Image dimensions used throughout the model
#############################################
IMG_H, IMG_W = 40, 300


#############################################
# CTC Decode Function (Greedy decoding)
# Collapses repeats and removes CTC blank token (0)
#############################################
def decode(preds):
    """
    Convert CTC network output to readable text.
    Args:
        preds (Tensor): model logits (B, T, C)
    Returns:
        list[str]: decoded strings
    """
    preds = preds.argmax(2).cpu().numpy()
    texts = []

    for seq in preds:
        s = ""
        prev = -1
        for p in seq:
            # Skip repeated characters and blanks (0)
            if p != prev and p != 0:
                s += idx2char[p]
            prev = p
        texts.append(s)
    return texts


#############################################
# Character Error Rate (CER)
# Lower CER = better model performance
#############################################
def cer(ref, hyp):
    """
    Compute Character Error Rate using sequence matching.
    Args:
        ref: ground truth string
        hyp: predicted string
    Returns:
        float: error rate (0 = perfect match)
    """
    from difflib import SequenceMatcher
    return 1 - SequenceMatcher(None, ref, hyp).ratio()


#############################################
# CAPTCHA Dataset Loader
# Supports mild augmentation during training
#############################################
class CaptchaDS(Dataset):
    def __init__(self, paths, labels, aug=False):
        """
        Args:
            paths (list[str]): image file paths
            labels (list[str]): corresponding ground-truth text
            aug (bool): enable training augmentation
        """
        if aug:  # Light augmentation to improve generalization
            self.t = T.Compose([
                T.Resize((IMG_H, IMG_W)), 
                T.ToTensor(),
                T.RandomApply([
                    T.RandomAffine(
                        degrees=4, 
                        shear=6, 
                        translate=(0.05, 0.05)
                    )
                ], p=0.2)
            ])
        else:   # Validation / inference mode (no augmentation)
            self.t = T.Compose([
                T.Resize((IMG_H, IMG_W)), 
                T.ToTensor()
            ])

        self.p = paths
        self.l = labels

    def __getitem__(self, i):
        """
        Load and preprocess 1 image + numeric label sequence
        Used in DataLoader batches
        """
        img = Image.open(self.p[i]).convert("RGB")
        img = self.t(img)

        # Convert label string → sequence of indices
        lbl = torch.tensor([char2idx[c] for c in self.l[i]], dtype=torch.long)
        return img, lbl

    def __len__(self):
        return len(self.p)
