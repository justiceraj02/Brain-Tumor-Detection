"""
Brain Tumor Detection — Flask Web Application
Classifies brain MRI scans into 4 categories using a fine-tuned ResNet50.

Author: Justice Raj
"""

import os
from io import BytesIO
from pathlib import Path

import flask
from flask import render_template, request

import torch
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from PIL import Image
from torchvision.transforms import v2
from torchvision.models import resnet50

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bt_resnet50_model.pt"

app = flask.Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
LABELS = ["None", "Meningioma", "Glioma", "Pituitary"]

# ---------------------------------------------------------------------------
# Device & Model Setup
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

resnet_model = resnet50(weights=None)
n_inputs = resnet_model.fc.in_features
resnet_model.fc = Sequential(
    Linear(n_inputs, 2048),
    SELU(),
    Dropout(p=0.4),
    Linear(2048, 2048),
    SELU(),
    Dropout(p=0.4),
    Linear(2048, 4),
    LogSigmoid(),
)

resnet_model.to(device)

if MODEL_PATH.exists():
    resnet_model.load_state_dict(
        torch.load(
            MODEL_PATH, map_location=torch.device(device), weights_only=True
        )
    )
    resnet_model.eval()
    MODEL_LOADED = True
else:
    MODEL_LOADED = False
    print(f"[WARNING] Model file not found at {MODEL_PATH}")
    print(
        "   Download it from: "
        "https://drive.google.com/file/d/1LJG_ITCWWtriLC5NPrWxIDwekWbhU_Rj/view"
    )
    print("   Place it in the 'models/' directory to enable predictions.")

# ---------------------------------------------------------------------------
# Image Preprocessing  (torchvision v2 — replaces deprecated ToTensor)
# ---------------------------------------------------------------------------
preprocess = v2.Compose(
    [
        v2.Resize((512, 512)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Open raw bytes as a PIL image, apply transforms, return a batch tensor."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return preprocess(img).unsqueeze(0)


@torch.inference_mode()
def get_prediction(image_bytes: bytes) -> tuple:
    """Return (class_name, confidence_percentage)."""
    tensor = preprocess_image(image_bytes).to(device)
    output = resnet_model(tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, class_id = torch.max(probabilities, dim=1)
    return LABELS[int(class_id)], round(float(confidence) * 100, 1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def main():
    """Landing page with project information."""
    return render_template("Diseasedet.html")


@app.route("/uimg", methods=["GET", "POST"])
def uimg():
    """Upload page (GET) and prediction handler (POST)."""
    if request.method == "GET":
        return render_template("uimg.html")

    # --- POST: process the uploaded image ---
    file = request.files.get("file")

    if file is None or file.filename == "":
        return (
            render_template(
                "error.html",
                message="No file selected. Please upload an MRI scan.",
            ),
            400,
        )

    if not allowed_file(file.filename):
        return (
            render_template(
                "error.html",
                message="Invalid file type. Please upload a PNG, JPG, or JPEG image.",
            ),
            400,
        )

    if not MODEL_LOADED:
        return (
            render_template(
                "error.html",
                message=(
                    "Model not loaded. Please download the model file and "
                    "place it in the 'models/' directory."
                ),
            ),
            503,
        )

    try:
        img_bytes = file.read()
        class_name, confidence = get_prediction(img_bytes)
        return render_template(
            "pred.html", result=class_name, confidence=confidence
        )
    except Exception as exc:
        return (
            render_template(
                "error.html",
                message=f"Error processing image: {exc}",
            ),
            500,
        )


@app.errorhandler(500)
def server_error(error):
    """Handle internal server errors."""
    return (
        render_template(
            "error.html", message="Internal server error. Please try again."
        ),
        500,
    )


@app.errorhandler(413)
def too_large(error):
    """Handle file-too-large errors."""
    return (
        render_template(
            "error.html", message="File too large. Maximum size is 16 MB."
        ),
        413,
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 62000))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)