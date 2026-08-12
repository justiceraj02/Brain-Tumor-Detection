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


def download_model():
    """Download the model file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("  Installing gdown...")
        import subprocess
        subprocess.check_call(["pip", "install", "gdown"])
        import gdown

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇️  Downloading model to {MODEL_PATH} ...")
    gdown.download(DRIVE_DIRECT, str(MODEL_PATH), quiet=False)


def check_model():
    """Check if the model file exists; download it if missing."""
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
        print("  Attempting automatic download...")
        print()
        try:
            download_model()
            if MODEL_PATH.exists():
                size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
                print(f"\n  \u2705 Download complete! Size: {size_mb:.1f} MB")
            else:
                print("\n  \u274c Download may have failed. Manual steps:")
                print(f"     1. Open: {DRIVE_URL}")
                print("     2. Click download in Google Drive")
                print(f"     3. Save as: {MODEL_PATH}")
        except Exception as e:
            print(f"\n  \u274c Auto-download failed: {e}")
            print()
            print("  Manual download steps:")
            print(f"     1. Open: {DRIVE_URL}")
            print("     2. Click download in Google Drive")
            print(f"     3. Save as: {MODEL_PATH}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    check_model()
