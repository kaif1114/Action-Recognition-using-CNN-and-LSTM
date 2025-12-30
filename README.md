# Action Recognition System with CNN-LSTM

A complete deep learning-based action recognition system with ResNet101 encoder, LSTM decoder with spatial attention, FastAPI backend, and web-based frontend. The system recognizes 40 different human actions in images from the Stanford 40 Actions dataset.

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Installation](#installation)
- [Training](#training)
- [Inference & Evaluation](#inference--evaluation)
- [Backend API](#backend-api)
- [Frontend Application](#frontend-application)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## ✨ Features

### Model Features
- **Pre-trained CNN**: ResNet101 encoder pre-trained on ImageNet for feature extraction
- **LSTM with Spatial Attention**: Bahdanau attention mechanism for interpretable predictions
- **Attention Visualization**: Generate 7×7 attention heatmaps showing where the model focuses
- **GPU Acceleration**: CUDA support for fast training and inference

### Application Features
- **FastAPI Backend**: RESTful API for real-time action prediction
- **Web Frontend**: Modern, responsive UI for image upload and result visualization
- **40 Action Classes**: Trained on Stanford 40 Actions Dataset (9,532 images)
- **Top-K Predictions**: Get multiple predictions with confidence scores
- **Real-time Processing**: 50-100ms inference time on GPU

---

## 🚀 Quick Start

Get your action recognition system running in 4 steps!

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- Stanford 40 Actions dataset in `dataset/` directory

### Step 1: Install Dependencies (5 minutes)

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Build Action Labels (< 1 minute)

```bash
python scripts/build_action_labels.py
```

**Output**: `checkpoints/action_labels.pkl` (40 action classes)

### Step 3: Train Model (30-35 hours)

```bash
python run_action_training.py
```

**What to Expect**:
- Training on 3,200 images, validating on 800 images
- ~40 minutes per epoch on RTX 3070 Laptop GPU (8GB)
- Automatic checkpointing and early stopping
- Expected accuracy: 75-80% after 50 epochs

**Quick Test** (2-3 hours):
Edit `run_action_training.py` and set `num_epochs: 3` for a quick functionality test.

### Step 4: Test & Deploy (5 minutes)

**Test Prediction**:
```bash
python inference/predict_action.py \
    --checkpoint checkpoints/actions/best_model.pth \
    --image dataset/JPEGImages/reading_001.jpg \
    --top_k 5
```

**Start Backend API**:
```bash
cd backend
python main_actions.py
```

**Start Frontend**:
```bash
cd frontend
python -m http.server 8080
```

Then open:
- **Frontend**: http://localhost:8080
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 🏗️ Architecture

### Model Architecture

```
Input Image (224×224×3)
    ↓
ResNet101 Encoder (pre-trained on ImageNet)
    ↓
Spatial Features (49×512) - 7×7 spatial grid
    ↓
LSTM with Bahdanau Attention (4 steps)
    ├── Step 1: Attend to spatial features
    ├── Step 2: Attend to spatial features
    ├── Step 3: Attend to spatial features
    └── Step 4: Attend to spatial features
    ↓
Classification Head (512 → 40)
    ↓
Action Prediction + Attention Heatmap (7×7)
```

### Encoder
- **Backbone**: ResNet101 (pre-trained on ImageNet)
- **Output**: 49 spatial features (7×7 grid) of 512 dimensions each
- **Fine-tuning**: Full fine-tuning enabled for better action-specific features

### Decoder
- **Type**: LSTM with Bahdanau Spatial Attention
- **LSTM Steps**: Fixed 4 steps (vs. variable-length for captioning)
- **Hidden Size**: 512 dimensions
- **Attention**: Computed over 7×7 spatial grid for each LSTM step
- **Output**: Classification logits for 40 action classes
- **Attention Gate**: Soft gating mechanism to control attention usage

### Key Difference from Image Captioning
Unlike image captioning which generates variable-length sequences, this model:
- Takes only images as input (no word embeddings)
- Runs LSTM for fixed number of steps
- Outputs single classification instead of sequences
- Returns attention weights for visualization

### Model Statistics
- **Total Parameters**: ~47M
- **Trainable Parameters**: ~45.5M
- **Model File Size**: ~190 MB

---

## 📦 Installation

### System Requirements

**Software**:
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Windows, Linux, or macOS

### Detailed Installation

1. **Clone Repository** (or download):
   ```bash
   cd E:\DL_Assignment04
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate Virtual Environment**:

   **Windows**:
   ```bash
   venv\Scripts\activate
   ```

   **Linux/Mac**:
   ```bash
   source venv/bin/activate
   ```

4. **Install PyTorch with CUDA**:
   ```bash
   # For CUDA 11.8 (check your CUDA version with nvidia-smi)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Install Other Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Verify Installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   ```

### Download Dataset

Download the Stanford 40 Actions dataset from:
http://vision.stanford.edu/Datasets/40actions.html

Extract to `dataset/` directory.

---

## 🎯 Training

### Prepare Action Labels

Before training, build the action label encoder:

```bash
python scripts/build_action_labels.py
```

This creates `checkpoints/action_labels.pkl` with mappings for 40 action classes.

### Train the Model

```bash
python run_action_training.py
```

### Training Configuration

Default hyperparameters (edit in `run_action_training.py`):

```python
config = {
    'batch_size': 32,                    # Optimized for 8GB GPU
    'num_epochs': 50,
    'learning_rate_encoder': 1e-4,       # Lower LR for pre-trained encoder
    'learning_rate_decoder': 1e-3,       # Higher LR for decoder
    'weight_decay': 1e-5,
    'encoder_dim': 512,
    'decoder_dim': 512,
    'lstm_steps': 4,
    'dropout': 0.5,
    'gradient_clip': 5.0,
    'early_stopping_patience': 10
}
```

### Dataset Splits
- **Training**: 3,200 images (80% of original train set)
- **Validation**: 800 images (20% of original train set)
- **Test**: 5,532 images (separate test set)

### Training Progress

**Expected Timeline**:
- Epoch 10: ~50-60% accuracy
- Epoch 30: ~70-75% accuracy
- Epoch 50: ~75-80% accuracy, ~90-95% top-5 accuracy

**Training Time** (RTX 3070 Laptop GPU):
- Per epoch: ~40 minutes
- Total (50 epochs): ~33 hours
- Early stopping may reduce total time

### Monitor Training

The training script displays:
- Real-time progress bars with loss and accuracy
- Validation metrics (accuracy, precision, recall, F1, top-5 accuracy)
- Automatic checkpoint saving
- Best model tracking

**Checkpoints saved to**:
- Latest: `checkpoints/actions/checkpoint_epoch_N.pth`
- Best: `checkpoints/actions/best_model.pth`

### Resume Training

If training is interrupted:

```bash
python run_action_training.py
# Prompt: "Resume training from checkpoint? (y/n)"
# Enter 'y' to continue from where you left off
```

---

## 🔮 Inference & Evaluation

### Single Image Prediction

```bash
python inference/predict_action.py \
    --checkpoint checkpoints/actions/best_model.pth \
    --image dataset/JPEGImages/reading_001.jpg \
    --top_k 5
```

**Output**:
```
Predicted action: reading
Confidence: 0.8523

Top-5 predictions:
  1. reading: 0.8523
  2. texting_message: 0.0654
  3. using_a_computer: 0.0321
  4. writing_on_a_book: 0.0198
  5. phoning: 0.0145

Visualizations saved to outputs/predictions
```

### Batch Prediction

```bash
python inference/predict_action.py \
    --checkpoint checkpoints/actions/best_model.pth \
    --image_dir dataset/JPEGImages \
    --output_dir outputs/predictions
```

Generates visualizations with:
- Original image
- Attention heatmap overlay
- Top-5 predictions with confidences

### Comprehensive Evaluation

Run evaluation on test set (5,532 images):

```bash
python scripts/evaluate_actions.py
```

**Generated Reports**:
- `outputs/evaluation/test_metrics.json` - Accuracy, precision, recall, F1
- `outputs/evaluation/confusion_matrix.png` - 40×40 confusion matrix
- `outputs/evaluation/per_class_accuracy.png` - Per-class accuracy chart
- `outputs/evaluation/top_mistakes.png` - Top-10 confusion pairs
- `outputs/evaluation/classification_report.txt` - Detailed metrics

---

## 🌐 Backend API

### Start API Server

```bash
cd backend
python main_actions.py
```

Or using uvicorn:
```bash
uvicorn backend.main_actions:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints**:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **List Actions**: http://localhost:8000/api/actions

### API Endpoints

#### POST /api/predict

Predict action for an uploaded image.

**Parameters**:
- `file` (required): Image file (JPEG, PNG, BMP, GIF, TIFF, WebP)
- `top_k` (optional): Number of top predictions (1-10, default: 5)
- `include_attention` (optional): Include attention heatmap (default: false)

**Example Request (curl)**:
```bash
curl -X POST "http://localhost:8000/api/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg" \
     -F "top_k=5" \
     -F "include_attention=true"
```

**Example Request (Python)**:
```python
import requests

url = "http://localhost:8000/api/predict"
files = {"file": open("test_image.jpg", "rb")}
params = {"top_k": 5, "include_attention": True}

response = requests.post(url, files=files, params=params)
result = response.json()

print(f"Action: {result['action']}")
print(f"Confidence: {result['confidence']:.3f}")
```

**Response**:
```json
{
  "action": "reading",
  "confidence": 0.8523,
  "top_k": [
    {"action": "reading", "confidence": 0.8523},
    {"action": "texting_message", "confidence": 0.0654},
    {"action": "using_a_computer", "confidence": 0.0321}
  ],
  "attention_heatmap": [[0.1, 0.2, ...], ...],  // 7×7 grid if requested
  "processing_time": 0.089,
  "device": "cuda:0"
}
```

#### GET /api/actions

List all 40 available action classes.

**Response**:
```json
{
  "num_classes": 40,
  "actions": ["applauding", "blowing_bubbles", "brushing_teeth", ...]
}
```

#### GET /health

Check API health status and model loading.

**Response**:
```json
{
  "status": "healthy",
  "device": "cuda:0",
  "model_loaded": true,
  "cuda_available": true,
  "num_classes": 40
}
```

---

## 🎨 Frontend Application

### Overview

A modern, responsive web frontend for the Action Recognition API with drag-and-drop upload, real-time predictions, and attention heatmap visualization.

### Features

- ✅ **Image Upload**: Drag-and-drop or file picker
- ✅ **Live Preview**: Image preview with metadata (filename, dimensions, size)
- ✅ **Prediction Display**: Top prediction with confidence color coding
- ✅ **Top-K Predictions**: Visual bar charts for all predictions
- ✅ **Attention Heatmap**: Interactive visualization with three view modes
- ✅ **Advanced Options**: Configurable top-k and heatmap settings
- ✅ **Health Status**: Real-time API connection indicator
- ✅ **Responsive Design**: Works on desktop, tablet, and mobile
- ✅ **Error Handling**: Clear, user-friendly error messages

### Start Frontend

#### Option 1: Python HTTP Server (Recommended)
```bash
cd frontend
python -m http.server 8080
```
Open http://localhost:8080

#### Option 2: Direct File Access
```bash
cd frontend
start index.html  # Windows
open index.html   # macOS
```

#### Option 3: Node.js HTTP Server
```bash
cd frontend
npx http-server -p 8080
```

### Using the Frontend

1. **Upload Image**:
   - Drag & drop image onto upload area, OR
   - Click "Choose File" button
   - Supported: JPEG, PNG, BMP, GIF, TIFF, WebP (max 10MB)

2. **Configure Options** (optional):
   - Click "⚙ Advanced Options"
   - Adjust Top-K slider (1-10)
   - Enable attention heatmap

3. **Predict Action**:
   - Click "🔍 Predict Action"
   - View results with confidence scores

4. **Explore Heatmap** (if enabled):
   - Switch view modes: Overlay | Original | Heatmap
   - Adjust opacity slider
   - Download heatmap as PNG

### Frontend File Structure

```
frontend/
├── index.html              # Main application
├── README.md              # Frontend documentation
├── css/
│   ├── main.css           # Design system & base styles
│   ├── components.css     # Component styles
│   └── responsive.css     # Mobile/tablet responsiveness
├── js/
│   ├── app.js            # Main controller
│   ├── api.js            # Backend communication
│   ├── ui.js             # UI manipulation
│   ├── visualization.js  # Heatmap rendering
│   └── utils.js          # Helper functions
└── assets/               # Optional assets
```

### Attention Heatmap Visualization

The frontend renders attention heatmaps using Canvas API:

1. **Receives** 7×7 attention grid from backend
2. **Interpolates** to image dimensions using bilinear interpolation
3. **Colors** using gradient: Blue (low) → Green → Yellow → Red (high)
4. **Three View Modes**:
   - **Overlay**: Original image + heatmap
   - **Original**: Image only
   - **Heatmap**: Attention visualization only
5. **Interactive Controls**: Opacity slider, download as PNG

### Configuration

**API Endpoint**: Click API endpoint in header to change (default: `http://localhost:8000`)

Settings are saved to browser localStorage.

---

## 📁 Project Structure

```
DL_Assignment04/
├── data/                          # Data loading and preprocessing
│   ├── action_labels.py           # Action label encoder (40 classes)
│   ├── stanford40_dataset.py      # Stanford 40 dataset loader
│   ├── transforms.py              # Image transformations
│   └── __init__.py
├── models/                        # Model architectures
│   ├── encoder.py                 # ResNet101 encoder
│   ├── action_decoder.py          # LSTM decoder for classification
│   ├── attention.py               # Spatial attention mechanism
│   ├── model.py                   # ActionRecognitionModel
│   └── __init__.py
├── training/                      # Training and validation
│   ├── train.py                   # Training functions
│   ├── validate_actions.py        # Classification validation
│   └── __init__.py
├── inference/                     # Action prediction utilities
│   ├── predict_action.py          # Prediction with visualization
│   └── __init__.py
├── backend/                       # FastAPI backend
│   ├── main_actions.py            # API for action recognition
│   └── __init__.py
├── frontend/                      # Web frontend
│   ├── index.html                 # Main application
│   ├── css/                       # Stylesheets
│   ├── js/                        # JavaScript modules
│   └── README.md                  # Frontend docs
├── scripts/                       # Utility scripts
│   ├── build_action_labels.py    # Build action labels
│   └── evaluate_actions.py       # Comprehensive evaluation
├── dataset/                       # Stanford 40 Actions dataset
│   ├── JPEGImages/                # 9,532 action images
│   ├── ImageSplits/               # Train/test split files
│   └── ...
├── checkpoints/                   # Model checkpoints
│   ├── action_labels.pkl          # Saved action labels
│   └── actions/                   # Training checkpoints
├── outputs/                       # Inference outputs (created)
│   ├── predictions/               # Prediction visualizations
│   └── evaluation/                # Evaluation results
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── run_action_training.py         # Main training script
└── README.md                      # This file
```

---

## 📊 Performance

### Expected Training Metrics

| Epoch | Accuracy | Top-5 Accuracy | F1 Score |
|-------|----------|----------------|----------|
| 10    | 50-60%   | 75-85%         | 0.50-0.60|
| 30    | 70-75%   | 85-92%         | 0.70-0.75|
| 50    | 75-80%   | 90-95%         | 0.75-0.80|

### Final Test Set Performance

- **Accuracy**: 75-80%
- **Top-5 Accuracy**: 90-95%
- **Precision/Recall/F1**: ~0.75-0.80 (weighted average)

### Inference Speed

**GPU (RTX 3070)**:
- Single image: 50-100ms
- With attention: +5-10ms
- Batch (32 images): ~30-40ms per image

**CPU**:
- Single image: 500-1000ms
- Batch processing recommended for efficiency

### Training Time

**RTX 3070 Laptop GPU (8GB)**:
- Per batch: ~20-25 seconds
- Per epoch: ~40 minutes
- Full training (50 epochs): ~33 hours
- Early stopping: ~20-30 epochs (~13-20 hours)

---

## 🐛 Troubleshooting

### CUDA Out of Memory

**Problem**: Training crashes with CUDA OOM error.

**Solutions**:
1. Reduce batch size:
   ```python
   'batch_size': 16,  # or 8 instead of 32
   ```
2. Reduce number of workers:
   ```python
   'num_workers': 0,
   ```
3. Clear CUDA cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Model Not Loading in API

**Problem**: Backend shows "Model not loaded" error.

**Solutions**:
1. Ensure model checkpoint exists: `checkpoints/actions/best_model.pth`
2. Ensure action labels exist: `checkpoints/action_labels.pkl`
3. Run label builder: `python scripts/build_action_labels.py`
4. Check backend logs for detailed error messages

### Slow Training

**Problem**: Training takes too long.

**Solutions**:
1. Verify GPU is being used:
   ```bash
   nvidia-smi  # Should show high GPU utilization
   ```
2. Enable CUDA optimizations (already in code):
   ```python
   torch.backends.cudnn.benchmark = True
   ```
3. Reduce validation frequency in `run_action_training.py`
4. Use num_workers > 0 on Linux (not recommended on Windows)

### Frontend Cannot Connect to API

**Problem**: "Cannot connect to API" error in frontend.

**Solutions**:
1. Ensure backend is running: `python backend/main_actions.py`
2. Check API endpoint in frontend header (default: `http://localhost:8000`)
3. Verify backend is accessible: Open `http://localhost:8000/docs` in browser
4. Check firewall settings
5. Try `http://127.0.0.1:8000` instead of `localhost`

### Dataset Not Found

**Problem**: Training fails with "Dataset not found" error.

**Solutions**:
1. Download Stanford 40 Actions dataset from http://vision.stanford.edu/Datasets/40actions.html
2. Extract to `dataset/` directory
3. Verify structure:
   ```
   dataset/
   ├── JPEGImages/
   └── ImageSplits/
       └── actions.txt
   ```

### Slow Inference

**Problem**: Predictions take a long time.

**Solutions**:
1. Check if using GPU: Look for "cuda:0" in results (vs "cpu")
2. Disable attention heatmap for faster predictions
3. Use batch processing for multiple images
4. Ensure model is in eval mode (already handled in code)

### Import Errors

**Problem**: Python cannot find modules.

**Solutions**:
1. Ensure virtual environment is activated
2. Install dependencies: `pip install -r requirements.txt`
3. Run from project root: `cd E:\DL_Assignment04`
4. Check Python version: `python --version` (requires 3.8+)

---

## 🔢 Action Classes (40 Total)

The Stanford 40 Actions dataset includes:

```
applauding, blowing_bubbles, brushing_teeth, cleaning_the_floor,
climbing, cooking, cutting_trees, cutting_vegetables, drinking,
feeding_a_horse, fishing, fixing_a_bike, fixing_a_car, gardening,
holding_an_umbrella, jumping, looking_through_a_microscope,
looking_through_a_telescope, phoning, playing_guitar, playing_violin,
pouring_liquid, pushing_a_cart, reading, riding_a_bike, riding_a_horse,
rowing_a_boat, running, shooting_an_arrow, smoking, taking_photos,
texting_message, throwing_frisby, using_a_computer, walking_the_dog,
washing_dishes, watching_TV, waving_hands, writing_on_a_board,
writing_on_a_book
```

---

## 📚 References

1. **Show, Attend and Tell**: Neural Image Caption Generation with Visual Attention
   - Kelvin Xu et al., ICML 2015
   - https://arxiv.org/abs/1502.03044
   - (We adapted the attention mechanism for classification)

2. **Stanford 40 Actions Dataset**
   - Bangpeng Yao et al., CVPR 2011
   - http://vision.stanford.edu/Datasets/40actions.html

3. **ResNet**: Deep Residual Learning for Image Recognition
   - Kaiming He et al., CVPR 2016
   - https://arxiv.org/abs/1512.03385

4. **FastAPI**: Modern, fast web framework for building APIs
   - https://fastapi.tiangolo.com

---

## 📄 License

This project is for educational purposes. The Stanford 40 Actions dataset has its own license terms.

---

## 🙏 Acknowledgments

- Stanford Vision Lab for the Stanford 40 Actions dataset
- PyTorch team for the deep learning framework
- FastAPI framework for the REST API
- Pre-trained ResNet models from torchvision

---

## 📧 Support

For issues and questions:
1. Check the Troubleshooting section above
2. Review code comments in each module
3. Test individual components:
   ```bash
   # Test data loading
   python -m data.stanford40_dataset

   # Test model creation
   python -m models.action_decoder
   ```

---

**Last Updated**: December 30, 2025
**Version**: 2.0.0 - Complete System Release
