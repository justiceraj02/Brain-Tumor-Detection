"""
Model Download Helper — Brain Tumor Detection

Run this script to check if the pre-trained ResNet50 model weights
are present and get instructions to download them if missing.

Author: Justice Raj
"""

from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "bt_resnet50_model.pt"
DRIVE_URL = (
    "https://drive.google.com/file/d/"
    "1LJG_ITCWWtriLC5NPrWxIDwekWbhU_Rj/view?usp=sharing"
)
DRIVE_DIRECT = (
    "https://drive.google.com/uc?export=download"
    "&id=1LJG_ITCWWtriLC5NPrWxIDwekWbhU_Rj"
)


def check_model():
    """Check if the model file exists and provide download instructions."""
    print("=" * 60)
    print("  \U0001f9e0 Brain Tumor Detection \u2014 Model Setup")
    print("=" * 60)
    print()

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"  \u2705 Model file found: {MODEL_PATH}")
        print(f"     Size: {size_mb:.1f} MB")
        print()
        print("  The model is ready. Run 'python app.py' to start the app.")
    else:
        print(f"  \u274c Model file NOT found at: {MODEL_PATH}")
        print()
        print("  To download the model, follow these steps:")
        print()
        print("  1. Open this link in your browser:")
        print(f"     {DRIVE_URL}")
        print()
        print("  2. Click the download button in Google Drive.")
        print()
        print("  3. Save the file as:")
        print(f"     {MODEL_PATH}")
        print()
        print("  Alternatively, if you have 'gdown' installed:")
        print("     pip install gdown")
        print(f"     gdown {DRIVE_DIRECT} -O \"{MODEL_PATH}\"")
        print()

    print("=" * 60)


if __name__ == "__main__":
    check_model()
