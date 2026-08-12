<div align="center">

# 🧠 Brain Tumor Detection — End to End

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch_2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask_3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![ResNet50](https://img.shields.io/badge/ResNet50-Transfer%20Learning-blueviolet?style=for-the-badge)](https://pytorch.org/vision/stable/models/resnet.html)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.3%25-brightgreen?style=for-the-badge)](#-model-performance)
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](../LICENSE.md)

> A full **end-to-end deep learning web application** that classifies brain tumors in MRI scans into four categories (No Tumor, Glioma, Meningioma, Pituitary) using a **fine-tuned ResNet50** via Transfer Learning — achieving **99.3% accuracy** — deployed as a Flask web app with a premium dark-mode UI.

**Developed by [Justice Raj](https://github.com/justiceraj02)**

</div>

---

## ⚠️ Medical Disclaimer

> **This tool is for educational and research purposes only.** It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or medical professional for clinical decisions.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [References & Citation](#-references--citation)

---

## 🔬 About the Project

Brain tumors are among the most critical conditions in medicine — accurate classification directly guides treatment decisions (surgery, radiation, chemotherapy). This project demonstrates how **Transfer Learning** with a pre-trained **ResNet50** can achieve near-perfect classification accuracy on MRI scans, far outperforming a CNN trained from scratch.

The model is trained on the **Jun Cheng Figshare brain tumor dataset** (3,064 T1-weighted CE-MRI images from 233 patients) and deployed as a **Flask web application** where users can upload an MRI image and receive a real-time classification with confidence score.

**What this project covers:**
- Converting `.mat` (MATLAB) MRI files to images and extracting tumor masks/borders
- Data augmentation with custom real-time transformations
- Fine-tuning ResNet50 with Transfer Learning using PyTorch
- Model evaluation with accuracy, loss curves, and per-class metrics
- Serving predictions via a Flask web app with a modern dark-mode UI

---

## ⚙️ How It Works

```
User Uploads MRI Scan (.jpg / .png)
              │
              ▼
    Image Preprocessing
  (Resize 512×512 → Normalize → Tensor)
              │
              ▼
   Fine-tuned ResNet50 Forward Pass
  (Custom classifier head → 4 classes)
              │
              ▼
     4-Class Prediction Output
  ┌────────┬────────────┬─────────────┬────────────┐
  │  None  │   Glioma   │ Meningioma  │ Pituitary  │
  └────────┴────────────┴─────────────┴────────────┘
              │
              ▼
  Predicted Class + Confidence Score
       Displayed in Browser
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | Brain Tumor Dataset |
| **Author** | Jun Cheng |
| **Source** | [Figshare — DOI: 10.6084/m9.figshare.1512427](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) |
| **Total Images** | 3,064 T1-weighted CE-MRI scans |
| **Patients** | 233 |
| **Format** | `.mat` (MATLAB) → converted to `.jpg` |
| **Task** | Multi-class classification (4 categories) |

### Class Distribution

| Class | Description | Slices |
|-------|-------------|:------:|
| ⚪ **None** | No tumor detected in the scan | — |
| 🔴 **Glioma** | Arises from glial cells; most common & aggressive brain tumor | 1,426 |
| 🟡 **Meningioma** | Grows on membranes surrounding the brain; often benign | 708 |
| 🟢 **Pituitary** | Forms on the pituitary gland at the brain's base; usually slow-growing | 930 |
| **Total** | | **3,064** |

### Data Augmentation

Custom real-time augmentations applied during training:

| Technique | Purpose |
|-----------|---------|
| Horizontal & Vertical Flip | Positional variance |
| Random Rotation (±15°) | Scan orientation variance |
| Brightness / Contrast Jitter | Scanner setting variance |
| Random Crop / Zoom | Variable tumor scale |
| Normalization (ImageNet μ/σ) | Stable gradient flow |

---

## 🏗️ Model Architecture

Rather than training a CNN from scratch, this project applies **Transfer Learning** by fine-tuning a **ResNet50** pretrained on ImageNet.

```
Input MRI Image (512 × 512 × 3)
          │
          ▼
┌──────────────────────────────────────┐
│        ResNet50 Backbone             │
│   (Pretrained on ImageNet)           │
│                                      │
│  Conv1 → BN → ReLU → MaxPool        │
│  Layer1: 3× Bottleneck blocks        │
│  Layer2: 4× Bottleneck blocks        │
│  Layer3: 6× Bottleneck blocks        │
│  Layer4: 3× Bottleneck blocks        │
│  Adaptive AvgPool → 2048-dim vector  │
└──────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│     Custom Classifier Head           │
│                                      │
│  FC (2048 → 2048) + SELU            │
│  Dropout (0.4)                       │
│  FC (2048 → 2048) + SELU            │
│  Dropout (0.4)                       │
│  FC (2048 → 4)                       │
│  LogSigmoid                          │
└──────────────────────────────────────┘
          │
          ▼
  None / Glioma / Meningioma / Pituitary
```

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Base Model | ResNet50 (ImageNet pretrained) |
| Strategy | Fine-tune full network after warm-up |
| Optimizer | Adam |
| Learning Rate | 0.001 (backbone), 0.01 (head) |
| LR Scheduler | StepLR (decay every 7 epochs) |
| Loss Function | Cross-Entropy Loss |
| Epochs | 25 |
| Batch Size | 32 |
| Train / Val / Test Split | 70% / 15% / 15% |

---

## 📈 Model Performance

| Metric | Score |
|--------|:-----:|
| **Overall Accuracy** | **~99.3%** |
| **Glioma F1** | ~99% |
| **Meningioma F1** | ~98% |
| **Pituitary F1** | ~99% |

> **Why Transfer Learning?** ResNet50 pre-trained on ImageNet already understands low-level features (edges, textures, shapes) that transfer well to MRI images. Fine-tuning requires far less data and training time while achieving significantly higher accuracy than training from scratch.

> Meningioma shows slightly lower scores due to class imbalance (708 vs 1,426 glioma images) and its high visual similarity to surrounding tissue.

---

## 📁 Project Structure

```
BRAIN TUMOR DETECTION [END 2 END]/
│
├── 📂 Brain-Tumor-Test-Images/         # Sample MRI images for testing
│   ├── 1.jpg, 2.jpg, ... 10.jpg
│
├── 📂 models/
│   ├── README.md                       # Download instructions for model weights
│   └── bt_resnet50_model.pt            # Fine-tuned ResNet50 weights (download separately)
│
├── 📂 static/
│   ├── b.jpg                           # Hero image for landing page
│   └── style.css                       # Custom dark-mode design system (CSS)
│
├── 📂 templates/
│   ├── base.html                       # Shared base template (navbar, footer, Bootstrap 5.3)
│   ├── Diseasedet.html                 # Landing page — project info & CTA
│   ├── uimg.html                       # Upload page — drag-and-drop MRI upload
│   ├── pred.html                       # Results page — classification + confidence
│   └── error.html                      # Error page — user-friendly error messages
│
├── app.py                              # Flask application entry point
├── setup_model.py                      # Helper script to check/download model weights
├── requirements.txt                    # Python dependencies (PyTorch, Flask, etc.)
├── .gitignore                          # Git ignore rules
└── README.md                           # Project documentation (you are here)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (recommended)
- **pip** package manager
- **GPU** (optional, CUDA-compatible for faster inference)

### 1. Clone the repository

```bash
git clone https://github.com/justiceraj02/first-project.git
cd "BRAIN TUMOR DETECTION [END 2 END]"
```

### 2. Set up virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the model weights

The trained ResNet50 model (~100 MB) is too large for Git. Download it manually:

```bash
# Option A: Run the setup helper
python setup_model.py

# Option B: Download directly
# Visit: https://drive.google.com/file/d/1LJG_ITCWWtriLC5NPrWxIDwekWbhU_Rj/view
# Save the file as: models/bt_resnet50_model.pt

# Option C: Using gdown
pip install gdown
gdown https://drive.google.com/uc?export=download&id=1LJG_ITCWWtriLC5NPrWxIDwekWbhU_Rj -O models/bt_resnet50_model.pt
```

### 5. Run the Flask app

```bash
python app.py
```

Navigate to → **http://127.0.0.1:5000** and upload an MRI scan.

> **Note:** The app will start even without the model file, but predictions will be disabled until the model is downloaded.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Deep Learning | PyTorch 2.x, Torchvision |
| Model | ResNet50 (Transfer Learning) |
| Image Processing | PIL (Pillow) |
| Web Framework | Flask 3.x |
| Frontend | HTML5, CSS3, Bootstrap 5.3.3, Jinja2 |
| Design | Custom dark-mode UI with glassmorphism |
| Model Serialization | `torch.save` / `.pt` |

---

## 📚 References & Citation

**Dataset — please cite if you use this work:**

```bibtex
@article{Cheng2015,
  author  = {Cheng, Jun and others},
  title   = {Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition},
  journal = {PLoS ONE},
  volume  = {10},
  number  = {10},
  year    = {2015}
}

@article{Cheng2016,
  author  = {Cheng, Jun and others},
  title   = {Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector Representation},
  journal = {PLoS ONE},
  volume  = {11},
  number  = {6},
  year    = {2016}
}
```

**Further reading:**
- [Jun Cheng Brain Tumor Dataset — Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- [Deep Residual Learning for Image Recognition — He et al. (2015)](https://arxiv.org/abs/1512.03385)
- [A survey on deep learning in medical image analysis — Litjens et al. (2017)](https://www.sciencedirect.com/science/article/pii/S1361841517301135)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

<div align="center">

Developed by [Justice Raj](https://github.com/justiceraj02)

⭐ Star this repo if it helped you!

</div>
