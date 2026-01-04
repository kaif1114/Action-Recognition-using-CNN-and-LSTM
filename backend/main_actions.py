"""FastAPI backend for action recognition."""
import os
import sys
import time
import io
from typing import Optional, List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch.nn.functional as F

from data.action_labels import ActionLabels
from data.vocabulary import Vocabulary
from data.transforms import get_inference_transform
from models.unified_model import load_unified_model


# Pydantic models
class PredictionDetail(BaseModel):
    """Individual prediction with action and confidence."""
    action: str
    confidence: float


class ActionResponse(BaseModel):
    """Response model for action prediction with caption."""
    action: str
    confidence: float
    caption: str
    top_k: List[PredictionDetail]
    attention_heatmap: Optional[List[List[float]]] = None
    processing_time: float
    device: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    device: str
    model_loaded: bool
    cuda_available: bool
    num_classes: int


# Global variables for model and labels
model = None
action_labels = None
vocabulary = None
device = None
transform = None


# FastAPI app
app = FastAPI(
    title="Action Recognition + Caption Generation API",
    description="API for recognizing human actions and generating captions for images using unified multi-task model with spatial attention",
    version="2.0.0"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (configure as needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model, action labels, and vocabulary on startup."""
    global model, action_labels, vocabulary, device, transform

    print("="*80)
    print("Starting Combined Action Recognition + Caption Generation API Server")
    print("="*80)

    # Paths - relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    action_checkpoint = os.path.join(base_dir, "checkpoints", "actions", "best_model.pth")
    caption_checkpoint = os.path.join(base_dir, "checkpoints", "captions", "best_model.pth")
    labels_path = os.path.join(base_dir, "checkpoints", "action_labels.pkl")
    vocabulary_path = os.path.join(base_dir, "checkpoints", "vocabulary.pkl")

    # Check if files exist
    if not os.path.exists(action_checkpoint):
        print(f"Warning: Action checkpoint not found at {action_checkpoint}")
        print("API will start but prediction will not work until model is available.")
        print(f"Please train the action model first using: python run_action_training.py")
        return

    if not os.path.exists(caption_checkpoint):
        print(f"Warning: Caption checkpoint not found at {caption_checkpoint}")
        print("API will start but caption generation will not work until model is available.")
        print(f"Please train the caption model first using: python run_caption_training.py")
        return

    if not os.path.exists(labels_path):
        print(f"Warning: Action labels not found at {labels_path}")
        print("API will start but action prediction will not work until labels are available.")
        print(f"Please build labels first using: python scripts/build_action_labels.py")
        return

    if not os.path.exists(vocabulary_path):
        print(f"Warning: Vocabulary not found at {vocabulary_path}")
        print("API will start but caption generation will not work until vocabulary is available.")
        print(f"Please build vocabulary first using: python scripts/build_vocabulary.py")
        return

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load action labels
    print(f"\nLoading action labels from: {labels_path}")
    action_labels = ActionLabels.load(labels_path)
    print(f"✓ Loaded {action_labels.num_classes} action classes")

    # Load vocabulary
    print(f"\nLoading vocabulary from: {vocabulary_path}")
    vocabulary = Vocabulary.load(vocabulary_path)
    print(f"✓ Loaded vocabulary with {len(vocabulary)} words")

    # Load unified model
    print(f"\nLoading unified model...")
    print(f"  Action checkpoint: {action_checkpoint}")
    print(f"  Caption checkpoint: {caption_checkpoint}")
    model = load_unified_model(action_checkpoint, caption_checkpoint, device)
    print("✓ Unified model loaded successfully")

    # Get inference transform
    transform = get_inference_transform()

    print("\n" + "="*80)
    print("API Server Ready!")
    print("  - Action Recognition: ✓")
    print("  - Caption Generation: ✓")
    print("  - Attention Visualization: ✓")
    print("="*80)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Action Recognition + Caption Generation API",
        "version": "2.0.0",
        "model": "Unified Multi-Task Model with Spatial Attention",
        "datasets": {
            "actions": "Stanford 40 Actions (40 classes)",
            "captions": "COCO Actions Subset (5004 vocab)"
        },
        "capabilities": [
            "Action Recognition",
            "Image Caption Generation",
            "Attention Visualization"
        ],
        "endpoints": {
            "/api/predict": "POST - Predict action and generate caption for an image",
            "/api/actions": "GET - List all action classes",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        device=str(device),
        model_loaded=model is not None,
        cuda_available=torch.cuda.is_available(),
        num_classes=action_labels.num_classes if action_labels else 0
    )


@app.get("/api/actions", tags=["Actions"])
async def list_actions():
    """List all available action classes."""
    if action_labels is None:
        raise HTTPException(
            status_code=503,
            detail="Action labels not loaded. Please check server logs."
        )

    actions = action_labels.get_all_actions()
    return {
        "num_classes": action_labels.num_classes,
        "actions": actions
    }


@app.post("/api/predict", response_model=ActionResponse, tags=["Prediction"])
async def predict_action(
    file: UploadFile = File(...),
    top_k: Optional[int] = 5,
    include_attention: Optional[bool] = False
):
    """
    Predict action and generate caption for an uploaded image.

    Args:
        file: Uploaded image file
        top_k: Number of top predictions to return (1-10)
        include_attention: Whether to include attention heatmap in response

    Returns:
        ActionResponse with predicted action, caption, and details
    """
    # Check if model, labels, and vocabulary are loaded
    if model is None or action_labels is None or vocabulary is None:
        raise HTTPException(
            status_code=503,
            detail="Model, action labels, or vocabulary not loaded. Please check server logs."
        )

    # Validate top_k
    if top_k < 1 or top_k > 10:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 10"
        )

    start_time = time.time()

    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading image file: {str(e)}"
        )

    try:
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predict combined (action + caption)
        model.eval()
        result = model.predict_combined(image_tensor, vocabulary, action_labels, device, top_k=top_k)

        processing_time = time.time() - start_time

        # Format top-k predictions
        top_k_predictions = [
            PredictionDetail(
                action=pred['action'],
                confidence=round(pred['confidence'], 4)
            )
            for pred in result['top_k']
        ]

        # Prepare attention heatmap if requested
        attention_heatmap = None
        if include_attention:
            attention_map = result['attention_map']
            attention_heatmap = attention_map.tolist()

        return ActionResponse(
            action=result['action'],
            confidence=round(result['confidence'], 4),
            caption=result['caption'],
            top_k=top_k_predictions,
            attention_heatmap=attention_heatmap,
            processing_time=round(processing_time, 3),
            device=str(device)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error predicting action: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "main_actions:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
