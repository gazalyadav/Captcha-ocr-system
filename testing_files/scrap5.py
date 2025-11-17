"""
scrap_ssc.py

Workflow:
1. Open https://ssc.gov.in/
2. Click “Login or Register”
3. Extract plaintext CAPTCHA inside <div class="captcha no-copy">
4. Save as text images (rendered) OR direct text labels (your choice)
5. Predict using CRNN (stage5_final.pth) for consistency with your pipeline

Requirements:
    pip install selenium pillow torch timm torchvision webdriver-manager
"""

import os
import time
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
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager


# ==================== CONFIG ====================
BASE_URL = "https://ssc.gov.in/"
SAVE_DIR = "/Users/gazalyadav_/Desktop/captcha/download5"
MODEL_PATH = "finetuned_ocr.pth"
NUM_CAPTCHAS = 10
TIMEOUT = 20

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)


# ==================== SELENIUM SETUP ====================
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


# =============== FUNCTION: MAKE TEXT INTO IMAGE (for OCR) ===============
def save_text_as_image(text, path):
    img = Image.new("RGB", (300, 40), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()

    draw.text((10, 5), text, font=font, fill="black")
    img.save(path)


# ==================== DOWNLOAD ONE CAPTCHA ====================
def download_captcha(i):
    driver.get(BASE_URL)
    print(f"[{i}] Opened SSC homepage.")

    # Step 1: Click “Login or Register”
    try:
        btn = wait.until(EC.element_to_be_clickable((
            By.XPATH, "//button[contains(text(),'Login') or contains(text(),'Register')]"
        )))
        driver.execute_script("arguments[0].click();", btn)
        print(f"[{i}] Clicked Login/Register.")
    except TimeoutException:
        raise RuntimeError("❌ Login/Register button not found.")

    # Step 2: Extract plaintext captcha from <div class="captcha no-copy">
    try:
        captcha_elem = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.captcha.no-copy"))
        )
        captcha_text = captcha_elem.text.strip()
        print(f"[{i}] Captcha found: {captcha_text}")
    except TimeoutException:
        raise RuntimeError("❌ Captcha <div> not found.")

    # Step 3: Save captcha text as an image (for your CRNN workflow)
    timestamp = int(time.time() * 1000)
    out_path = os.path.join(SAVE_DIR, f"ssc_captcha_{timestamp}.png")

    save_text_as_image(captcha_text, out_path)

    print(f"[{i}] Saved → {out_path}")
    return out_path, captcha_text


# ==================== DOWNLOAD MULTIPLE ====================
print(f"\n🔁 Downloading {NUM_CAPTCHAS} SSC CAPTCHAs...\n")
downloaded = []
ground_truth = []

for i in range(1, NUM_CAPTCHAS + 1):
    try:
        p, txt = download_captcha(i)
        downloaded.append(p)
        ground_truth.append(txt)
        time.sleep(1)
    except Exception as e:
        print(f"[{i}] ERROR: {e}")

driver.quit()
print("\n🔒 Browser closed.")


# ==================== LOAD CRNN MODEL ====================
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


transform = T.Compose([
    T.Resize((40, 300)),
    T.ToTensor()
])


def decode(preds):
    preds = preds.argmax(2).cpu().numpy()[0]
    txt, last = "", -1
    for p in preds:
        if p != 0 and p != last:
            txt += idx2char[p]
        last = p
    return txt


def predict(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return decode(model(img))


# ==================== PREDICT ALL ====================
print("\n🔍 Predicting SSC CAPTCHAs...\n")
for i, p in enumerate(downloaded):
    pred = predict(p)
    print(f"🧩 {os.path.basename(p)} → {pred}   (GT: {ground_truth[i]})")

print("\n🏁 Done.")
print("📂 Saved in:", SAVE_DIR)
