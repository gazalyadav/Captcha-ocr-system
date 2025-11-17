"""
scrap4.py
Broadcast Seva – Login Captcha Scraper + CRNN Prediction
(With fix for transparent CAPTCHA images)

Requirements:
    pip install selenium requests torch torchvision timm pillow webdriver-manager
"""

import os
import time
import requests
import string
import torch
import timm
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException


# ===========================================================
# CONFIG
# ===========================================================
BASE_URL = "https://new.broadcastseva.gov.in/digigov-portal-web-app/jsp/mib/common/login.jsp"
CAPTCHA_BASE = "https://new.broadcastseva.gov.in/digigov-portal-web-app/"
SAVE_DIR = "/Users/gazalyadav_/Desktop/captcha/download4"
MODEL_PATH = "stage5_final.pth"
NUM_CAPTCHAS = 10
TIMEOUT = 15

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)


# ===========================================================
# SELENIUM SETUP
# ===========================================================
chrome_opts = Options()
chrome_opts.add_argument("--no-sandbox")
chrome_opts.add_argument("--disable-gpu")
chrome_opts.add_argument("--disable-dev-shm-usage")
chrome_opts.add_argument("--window-size=1400,900")
# chrome_opts.add_argument("--headless=new")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=chrome_opts
)

wait = WebDriverWait(driver, TIMEOUT)


# ===========================================================
# FIX TRANSPARENT CAPTCHA → WHITE BACKGROUND
# ===========================================================
def fix_transparent_captcha(path):
    img = Image.open(path).convert("RGBA")
    new_img = Image.new("RGB", img.size, (255, 255, 255))  # white bg
    new_img.paste(img, mask=img.split()[3])                # apply alpha channel
    fixed_path = path.replace(".png", "_fixed.png")
    new_img.save(fixed_path)
    return fixed_path


# ===========================================================
# DOWNLOAD SINGLE CAPTCHA
# ===========================================================
def download_captcha(i):
    driver.get(BASE_URL)
    print(f"[{i}] Opened BroadcastSeva Login page.")

    # Wait for captcha
    try:
        img_elem = wait.until(
            EC.presence_of_element_located((By.ID, "captcha"))
        )
        print(f"[{i}] CAPTCHA element found.")
    except TimeoutException:
        raise RuntimeError("❌ CAPTCHA not found.")

    # Extract src
    src = img_elem.get_attribute("src")

    # Correct URL building
    if src.startswith("SimpleCaptcha"):
        src = CAPTCHA_BASE + src

    timestamp = int(time.time() * 1000)
    out_path = os.path.join(SAVE_DIR, f"broadcast_captcha_{timestamp}.png")

    # Download raw PNG
    try:
        r = requests.get(src, stream=True, timeout=10)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for ch in r.iter_content(1024):
                f.write(ch)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download captcha: {e}")

    # FIX transparency
    fixed_path = fix_transparent_captcha(out_path)
    print(f"[{i}] Saved FIXED captcha → {fixed_path}")

    return fixed_path


# ===========================================================
# DOWNLOAD MULTIPLE CAPTCHAS
# ===========================================================
print(f"\n🔁 Downloading {NUM_CAPTCHAS} Captchas from Broadcast Seva Login...\n")
downloaded_paths = []

for i in range(1, NUM_CAPTCHAS + 1):
    try:
        path = download_captcha(i)
        downloaded_paths.append(path)
        time.sleep(1)
    except Exception as e:
        print(f"[{i}] ERROR: {e}")

driver.quit()
print("\n🔒 Browser closed.\n")


# ===========================================================
# LOAD CRNN MODEL
# ===========================================================
CHARS = string.ascii_lowercase + string.ascii_uppercase + "0123456789"
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
idx2char[0] = ""

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🧠 Using device: {device.upper()}")


class CRNN(torch.nn.Module):
    def __init__(self, nc=len(CHARS) + 1):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.lstm = torch.nn.LSTM(self.backbone.num_features, 256, 2,
                                  batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(512, nc)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        f = f.mean(2).permute(0, 2, 1)
        x, _ = self.lstm(f)
        return self.fc(x).log_softmax(2)


model = CRNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Loaded OCR model: {MODEL_PATH}")

transform = T.Compose([
    T.Resize((40, 300)),
    T.ToTensor()
])


# ===========================================================
# OCR DECODE + PREDICTION
# ===========================================================
def decode(preds):
    preds = preds.argmax(2).cpu().numpy()[0]
    text, last = "", -1
    for p in preds:
        if p != 0 and p != last:
            text += idx2char[p]
        last = p
    return text


def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
    return decode(out)


# ===========================================================
# PREDICT ALL DOWNLOADED CAPTCHAS
# ===========================================================
print("🔍 Predicting downloaded Broadcast Seva Login CAPTCHAs...\n")

for p in sorted(downloaded_paths):
    try:
        pred = predict(p)
        print(f"🧩 {os.path.basename(p)} → {pred}")
    except Exception as e:
        print(f"⚠️ Error predicting {p}: {e}")

print("\n🏁 DONE.")
print(f"📂 Saved in: {SAVE_DIR}")
