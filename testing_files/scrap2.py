"""
scrap_predict_fixed.py
- Normalizes unwanted chars
- Tracks downloaded files this run and deletes them immediately if prediction matches
- Predicts only the final valid files (not every file in folder)
"""

import os
import time
import base64
import requests
import torch
import timm
import string
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# ==================== CONFIG ====================
DEMO_URL = "https://ireps.gov.in/"
SAVE_DIR = "/Users/gazalyadav_/Desktop/captcha/download2"
MODEL_PATH = "stage5_final.pth"
TIMEOUT = 15
TARGET_COUNT = 15
MAX_ATTEMPTS = 30

# original set (you can edit this). It will be normalized to lowercase below.
UNWANTED_CHARS = {"p", "9", "f", "E", "c", "e"}

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Normalize unwanted set to lowercase to avoid case mismatch bugs
UNWANTED = set(ch.lower() for ch in UNWANTED_CHARS)

# ==================== SETUP SELENIUM ====================
chrome_opts = Options()
chrome_opts.add_argument("--no-sandbox")
chrome_opts.add_argument("--disable-gpu")
chrome_opts.add_argument("--disable-dev-shm-usage")
chrome_opts.add_argument("--window-size=1200,900")
# chrome_opts.add_argument("--headless=new")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_opts)
wait = WebDriverWait(driver, TIMEOUT)

# ==================== LOAD MODEL ====================
CHARS = string.ascii_lowercase + string.ascii_uppercase + "0123456789"
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
idx2char[0] = ""
device = "mps" if torch.backends.mps.is_available() else "cpu"

class CRNN(torch.nn.Module):
    def __init__(self, nc=len(CHARS) + 1):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.lstm = torch.nn.LSTM(self.backbone.num_features, 256, 2, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(512, nc)

    def forward(self, x):
        f = self.backbone.forward_features(x)
        f = f.mean(2).permute(0, 2, 1)
        x, _ = self.lstm(f)
        return self.fc(x).log_softmax(2)

print(f"\n🧠 Loading CRNN model on {device.upper()}...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")
model = CRNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Model loaded successfully: {MODEL_PATH}")

transform = T.Compose([T.Resize((40, 300)), T.ToTensor()])

def decode(preds):
    preds = preds.argmax(2).cpu().numpy()[0]
    s, last = "", -1
    for p in preds:
        if p != 0 and p != last:
            s += idx2char[p]
        last = p
    return s

def predict_text(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
    return decode(logits)

def contains_unwanted(pred):
    # pred may be mixed case; normalize to lowercase for membership check
    if not pred:
        return False
    s = str(pred).lower()
    return any(ch in UNWANTED for ch in s)

# ==================== CAPTCHA DOWNLOAD ====================
def download_captcha():
    driver.get(DEMO_URL)
    # minimal logging
    print("Opened IREPS homepage.")

    # Close modal popup if present
    try:
        close_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-dismiss='modal']")))
        close_btn.click()
        print("Closed homepage popup.")
        time.sleep(0.4)
    except TimeoutException:
        pass

    # Click “Search E-Tenders”
    try:
        search_link = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(., 'Search E-Tenders')]")))
        driver.execute_script("arguments[0].click();", search_link)
        print("Clicked 'Search E-Tenders' → Page loading...")
    except TimeoutException:
        raise RuntimeError("Could not locate 'Search E-Tenders' link.")

    # Wait for captcha
    try:
        captcha_elem = wait.until(EC.presence_of_element_located((By.ID, "imgCaptcha")))
        wait.until(lambda d: captcha_elem.get_attribute("src"))
        print("Captcha image located.")
    except TimeoutException:
        raise RuntimeError("Captcha did not load properly.")

    src = captcha_elem.get_attribute("src")
    timestamp = int(time.time() * 1000)
    out_path = os.path.join(SAVE_DIR, f"captcha_{timestamp}.png")

    # Save image (data URI or URL)
    if src.startswith("data:image"):
        _, b64_data = src.split(",", 1)
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64_data))
    else:
        if src.startswith("/"):
            src = f"{DEMO_URL.rstrip('/')}/{src.lstrip('/')}"
        r = requests.get(src, stream=True, timeout=10)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

    print(f"✅ Captcha saved: {out_path}")
    return out_path

# ==================== MAIN LOGIC ====================
valid_files = []
downloaded_this_run = []  # track files downloaded during this run

print(f"\n🔁 Collecting up to {TARGET_COUNT} valid CAPTCHAs (max attempts {MAX_ATTEMPTS})...\n")

for _ in range(MAX_ATTEMPTS):
    if len(valid_files) >= TARGET_COUNT:
        break

    try:
        img_path = download_captcha()
        downloaded_this_run.append(img_path)

        pred = predict_text(img_path).strip()
        # delete immediately if unwanted (comparison uses lowercase UNWANTED set)
        if not pred or contains_unwanted(pred):
            try:
                os.remove(img_path)
                print(f"🗑️ Deleted (unwanted) -> {os.path.basename(img_path)}")
            except Exception:
                print(f"⚠️ Failed to delete {img_path}")
            continue

        # keep valid file
        valid_files.append(img_path)
        time.sleep(1.0)

    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(1)

print(f"\n✅ CAPTCHA collection complete. Valid images: {len(valid_files)}")
for f in valid_files:
    print(" -", os.path.basename(f))

# ==================== AUTO PREDICT ONLY VALID IMAGES ====================
print("\n🔍 Starting prediction on final valid CAPTCHAs...\n")

if not valid_files:
    print("❌ No valid CAPTCHA images to predict.")
else:
    for f in valid_files:
        try:
            pred = predict_text(f)
            print(f"🧩 {os.path.basename(f)} → {pred}")
        except Exception as e:
            print(f"⚠️ Error predicting {os.path.basename(f)}: {e}")

print("\n🏁 All predictions completed.")
driver.quit()
print("🔒 Driver closed. Done.")
