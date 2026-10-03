# 🤚 Hand Gesture Recognition using Deep Learning

A beginner-friendly deep learning project that classifies **14 different hand gestures** in real time using **TensorFlow/Keras** and **transfer learning with MobileNetV2**.

Built as a hands-on learning project for **computer vision** and **convolutional neural networks (CNNs)**.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19%2B-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Gesture Classes](#-gesture-classes)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Usage](#-usage)
- [Future Roadmap](#-future-roadmap)

---

## 🔍 Project Overview

| Aspect          | Detail                                                   |
| --------------- | -------------------------------------------------------- |
| **Goal**        | Classify 14 hand gestures from images                    |
| **Model**       | MobileNetV2 (ImageNet pretrained) via transfer learning  |
| **Framework**   | TensorFlow / Keras                                       |
| **Dataset**     | HG14 Hand Gesture Dataset (Kaggle)                       |
| **Input**       | RGB images resized to 224 × 224                          |
| **Inference**   | Single image prediction **and** real-time webcam feed    |

### Key Concepts Covered

- Transfer learning with a pretrained backbone
- Data augmentation to improve generalisation
- Fine-tuning vs. feature extraction
- Training callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
- Evaluation with confusion matrices and classification reports
- Real-time inference with OpenCV

---

## ✋ Gesture Classes

The HG14 dataset contains **14 hand gesture categories**:

| #  | Gesture        | #  | Gesture        |
| -- | -------------- | -- | -------------- |
| 1  | Doing Other    | 8  | Mute           |
| 2  | No Gesture     | 9  | OK             |
| 3  | Drag           | 10 | Palm           |
| 4  | Fist           | 11 | Pan            |
| 5  | Grab           | 12 | Point           |
| 6  | Like           | 13 | Stop           |
| 7  | Dislike        | 14 | Zoom           |

> **Note:** Exact class names will be confirmed once the dataset is downloaded. The table above is representative of the HG14 categories.

---

## 📁 Folder Structure

```
HandGestureRecognition/
│
├── dataset/                # HG14 dataset (downloaded from Kaggle)
│   ├── train/              #   Training images (one subfolder per gesture)
│   ├── val/                #   Validation images
│   └── test/               #   Test images
│
├── models/                 # Saved trained models (.keras files)
│
├── notebooks/              # Jupyter notebooks for EDA & experiments
│
├── outputs/                # Training curves, confusion matrices, logs
│
├── train.py                # Model training pipeline
├── predict.py              # Single-image prediction script
├── webcam.py               # Real-time webcam gesture recognition
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.11+** — [Download](https://www.python.org/downloads/)
- **pip** (comes with Python)
- **Git** — [Download](https://git-scm.com/)
- A **GPU** is recommended for training but not required

### Step-by-Step

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/HandGestureRecognition.git
   cd HandGestureRecognition
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify TensorFlow installation**

   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

---

## 📦 Dataset Setup

This project uses the **HG14 Hand Gesture Dataset** from Kaggle.

### Download

1. Visit the dataset page: [HG14 on Kaggle](https://www.kaggle.com/datasets/antobenedetti/hg14-hand-gesture-dataset)
2. Click **Download** (requires a free Kaggle account)

   *Or* use the Kaggle CLI:

   ```bash
   pip install kaggle
   kaggle datasets download -d antobenedetti/hg14-hand-gesture-dataset
   ```

### Place the Data

Extract the downloaded archive so the `dataset/` folder looks like this:

```
dataset/
├── train/
│   ├── gesture_class_1/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── gesture_class_2/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

> **Important:** Each gesture class should have its own subfolder inside `train/`, `val/`, and `test/`. If the Kaggle download has a different structure, reorganise it to match the layout above before training.

---

## 🚀 Usage

> ⚠️ The commands below will work once the corresponding modules are implemented.

### Train the Model

```bash
python train.py
```

### Predict on a Single Image

```bash
python predict.py --image path/to/hand_image.jpg
```

### Real-Time Webcam Recognition

```bash
python webcam.py
```

---

## 📚 Required Libraries

| Library        | Purpose                                        |
| -------------- | ---------------------------------------------- |
| TensorFlow     | Deep learning framework (includes Keras)       |
| OpenCV         | Image processing & webcam capture              |
| Pillow         | Image loading utilities                        |
| NumPy          | Numerical array operations                     |
| Matplotlib     | Plotting training curves                       |
| Seaborn        | Confusion matrix heatmaps                      |
| scikit-learn   | Classification metrics & evaluation            |

All versions are pinned in [`requirements.txt`](requirements.txt).

---

## 🗺️ Future Roadmap

- [x] **Phase 1** — Project scaffolding, README, and requirements
- [ ] **Phase 2** — Data loading, exploration, and augmentation (`train.py`)
- [ ] **Phase 3** — MobileNetV2 transfer learning model architecture
- [ ] **Phase 4** — Training pipeline with callbacks and checkpoints
- [ ] **Phase 5** — Evaluation: confusion matrix, classification report, training curves
- [ ] **Phase 6** — Single-image prediction (`predict.py`)
- [ ] **Phase 7** — Real-time webcam inference (`webcam.py`)
- [ ] **Phase 8** — Hyperparameter tuning & experiment tracking
- [ ] **Phase 9** — Model export (TFLite / SavedModel) for deployment
- [ ] **Phase 10** — Web demo with Gradio or Streamlit

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---