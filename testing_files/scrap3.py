"""
scrap_nivesh.py
End-to-end automation:
1. Automatically downloads 10 CAPTCHA images from the Nivesh Mitra portal.
2. Predicts text using your CRNN model (stage5_final.pth).

Requirements:
    pip install selenium requests torch torchvision timm pillow webdriver-manager
"""

import os
import time
import requests
import base64
import string
import torch
import timm
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
BASE_URL = "https://niveshmitra.up.nic.in/"
SAVE_DIR = "/Users/gazalyadav_/Desktop/captcha/download3"
MODEL_PATH = "stage5_final.pth"
TIMEOUT = 20
NUM_CAPTCHAS = 10

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)


# ==================== SETUP CHROME ====================
chrome_opts = Options()
chrome_opts.add_argument("--no-sandbox")
chrome_opts.add_argument("--disable-gpu")
chrome_opts.add_argument("--disable-dev-shm-usage")
chrome_opts.add_argument("--window-size=1200,900")
# chrome_opts.add_argument("--headless=new")  # Uncomment for headless mode

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_opts)
wait = WebDriverWait(driver, TIMEOUT)


# ==================== DOWNLOAD CAPTCHA FUNCTION ====================
def download_captcha():
    driver.get(BASE_URL)
    print("Opened Nivesh Mitra homepage.")

    # Close popup if it appears
    try:
        close_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "span.close-btn"))
        )
        driver.execute_script("arguments[0].click();", close_btn)
        print("Closed popup.")
    except TimeoutException:
        print("⚠️ No popup found — continuing...")

    # Click "Login"
    try:
        login_link = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, 'Login.aspx')]"))
        )
        driver.execute_script("arguments[0].click();", login_link)
        print("Clicked 'Login' → Login page opened.")
    except TimeoutException:
        raise RuntimeError("❌ Could not find Login link on the homepage.")

    # Handle “Got it!” popup
    try:
        got_it_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.swal2-confirm.swal2-styled"))
        )
        driver.execute_script("arguments[0].click();", got_it_btn)
        print("Dismissed 'Got it!' popup.")
    except TimeoutException:
        print("⚠️ No 'Got it!' popup — continuing...")

    # Wait for CAPTCHA
    try:
        captcha_elem = wait.until(
            EC.presence_of_element_located((By.ID, "MainContent_ImgCaptcha"))
        )
        wait.until(lambda d: captcha_elem.get_attribute("src"))
        print("Captcha image located.")
    except TimeoutException:
        raise RuntimeError("❌ Captcha not found on login page.")

    src = captcha_elem.get_attribute("src")
    timestamp = int(time.time())
    out_path = os.path.join(SAVE_DIR, f"captcha_{timestamp}.png")

    # Handle relative URL
    if src.startswith("/"):
        src = f"{BASE_URL.rstrip('/')}/{src.lstrip('/')}"

    try:
        r = requests.get(src, stream=True, timeout=10)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        print(f"✅ Captcha saved: {out_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download captcha: {e}")

    return out_path


# ==================== LOOP TO DOWNLOAD MULTIPLE CAPTCHAS ====================
print(f"\n🔁 Downloading {NUM_CAPTCHAS} CAPTCHA images from Nivesh Mitra...\n")
downloaded = []

for i in range(NUM_CAPTCHAS):
    try:
        path = download_captcha()
        downloaded.append(path)
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ Error ({i+1}/{NUM_CAPTCHAS}): {e}")

print(f"\n✅ Completed downloading {len(downloaded)} CAPTCHAs.")
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
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Loaded model: {MODEL_PATH}")

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
