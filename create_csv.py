"""
------------------------------------------------------------
CREATE CSV FROM CAPTCHA IMAGE FILENAMES
------------------------------------------------------------

This script scans a dataset folder, extracts CAPTCHA labels 
from image filenames (e.g., "aB3k9.jpg" → label: aB3k9),
and saves all filenames + labels into a CSV file.

This CSV can later be used for dataset indexing or external analysis.

Usage:
    python create_csv.py
"""

# ------------------------------------------------------------
# IMPORT LIBRARIES
# ------------------------------------------------------------

import os              # File and directory operations
import pandas as pd    # Pandas used to create and save dataset table

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# Folder containing CAPTCHA images
IMAGE_DIR = '/Users/gazalyadav_/Desktop/captcha/data/archive'

# Output CSV file path
OUTPUT_CSV_PATH = os.path.join(IMAGE_DIR, 'master_dataset.csv')


# ------------------------------------------------------------
# FUNCTION: CREATE CSV FROM IMAGE FILE NAMES
# ------------------------------------------------------------

def create_csv_from_filenames():
    """
    Reads all images from IMAGE_DIR, extracts labels from filenames,
    and stores them into a CSV in format:

        filename,label
        3fT9k.png,3fT9k
        aB92D.jpg,aB92D
        ...

    Assumes filename (without extension) = CAPTCHA label.
    """

    print("="*60)
    print(f"📂 Scanning dataset directory: {IMAGE_DIR}")
    print("="*60)

    # Validate directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ ERROR: Directory does not exist → {IMAGE_DIR}")
        return

    # Read all image files (.jpg/.png/.jpeg)
    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if len(image_files) == 0:
        print("❌ ERROR: No image files found. Dataset directory is empty.")
        return

    print(f"✅ Found {len(image_files)} image files.")

    dataset_records = []

    # --------------------------------------------------------
    # Loop through all images and extract labels
    # --------------------------------------------------------
    for filename in image_files:
        # Extract text before extension as label
        label = os.path.splitext(filename)[0]

        dataset_records.append({
            'filename': filename,
            'label': label
        })

    # Create DataFrame
    df = pd.DataFrame(dataset_records)

    # Save CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("\n✅ CSV creation successful!")
    print(f"📁 Saved dataset to: {OUTPUT_CSV_PATH}")
    print("="*60)


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------
if __name__ == '__main__':
    create_csv_from_filenames()
