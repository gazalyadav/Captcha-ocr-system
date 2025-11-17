"""
scrap_mhada.py
End-to-end automation:
1. Automatically downloads 10 CAPTCHA images from MHADA lottery portal.
2. Runs OCR prediction using CRNN model (stage5_final.pth).

Requirements:
    pip install selenium requests torch torchvision timm pillow webdriver-manager
"""

import os
import time
import base64
import requests
import torch
import timm
import string
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


# ==================== CONFIG ====================
BASE_URL = "https://lottery.mhada.gov.in/"
SAVE_DIR = "/Users/gazalyadav_/Desktop/captcha/download1"
MODEL_PATH = "stage5_final.pth"
NUM_CAPTCHAS = 10
TIMEOUT = 20

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# ==================== SETUP CHROME ====================
chrome_opts = Options()
chrome_opts.add_argument("--no-sandbox")
chrome_opts.add_argument("--disable-gpu")
chrome_opts.add_argument("--disable-dev-shm-usage")
chrome_opts.add_argument("--window-size=1200,900")
# chrome_opts.add_argument("--headless=new")  # Uncomment if you want headless mode

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_opts)
wait = WebDriverWait(driver, TIMEOUT)


# ==================== DOWNLOAD CAPTCHA FUNCTION ====================
def download_captcha():
    driver.get(BASE_URL)
    print("Opened MHADA homepage.")

    # Click "Go" button
    try:
        go_btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Go') and contains(@href, 'Konkan')]"))
        )
        driver.execute_script("arguments[0].click();", go_btn)
        print("Clicked 'Go' button → Konkan Lottery page opened.")
    except TimeoutException:
        raise RuntimeError("❌ 'Go' button not found on homepage.")

    # Click "Login" button
    try:
        login_btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(@data-target, '#loginModal')]"))
        )
        driver.execute_script("arguments[0].click();", login_btn)
        print("Clicked 'Login' → Login modal opened.")
    except TimeoutException:
        raise RuntimeError("❌ 'Login' button not found on page.")

    # Wait for CAPTCHA image
    try:
        captcha_elem = wait.until(EC.presence_of_element_located((By.ID, "captchaimg")))
        wait.until(lambda d: captcha_elem.get_attribute("src"))
        print("Captcha image loaded.")
    except TimeoutException:
        raise RuntimeError("❌ Captcha image not found or failed to load.")

    src = captcha_elem.get_attribute("src")
    timestamp = int(time.time())
    out_path = os.path.join(SAVE_DIR, f"captcha_{timestamp}.png")

    if src.startswith("data:image"):
        _, b64_data = src.split(",", 1)
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64_data))
    else:
        if src.startswith("/"):
            src = f"{BASE_URL.rstrip('/')}/{src.lstrip('/')}"
        try:
            r = requests.get(src, stream=True, timeout=10)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download captcha: {e}")

    print(f"✅ Saved: {out_path}")
    return out_path


# ==================== CAPTCHA DOWNLOAD LOOP ====================
print(f"\n🔁 Downloading {NUM_CAPTCHAS} CAPTCHA images from MHADA portal...\n")
downloaded = []
for i in range(NUM_CAPTCHAS):
    try:
        path = download_captcha()
        downloaded.append(path)
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ Error ({i+1}/{NUM_CAPTCHAS}): {e}")

print(f"\n✅ Completed download of {len(downloaded)} CAPTCHAs.")
driver.quit()
print("🔒 Browser closed.")


# ==================== LOAD CRNN MODEL ====================
CHARS = string.ascii_lowercase + string.ascii_uppercase + "0123456789"
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
idx2char[0] = ""
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\n🧠 Using device: {device.upper()}")

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

model = CRNN().to(device)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Loaded CRNN model from: {MODEL_PATH}")

transform = T.Compose([T.Resize((40, 300)), T.ToTensor()])

def decode(preds):
    preds = preds.argmax(2).cpu().numpy()[0]
    s, last = "", -1
    for p in preds:
        if p != 0 and p != last:
            s += idx2char[p]
        last = p
    return s

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
    return decode(logits)


# ==================== RUN PREDICTIONS ====================
print("\n🔍 Predicting downloaded CAPTCHAs...\n")

for img_path in sorted(downloaded):
    try:
        pred = predict(img_path)
        print(f"🧩 {os.path.basename(img_path)} → {pred}")
    except Exception as e:
        print(f"⚠️ Error predicting {os.path.basename(img_path)}: {e}")

print("\n🏁 Process complete.")
print(f"📂 Folder: {SAVE_DIR}")
