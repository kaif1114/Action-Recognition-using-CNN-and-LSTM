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
from data.transforms import get_inference_transform
from models.model import create_action_model


# Pydantic models
class PredictionDetail(BaseModel):
    """Individual prediction with action and confidence."""
    action: str
    confidence: float


class ActionResponse(BaseModel):
    """Response model for action prediction."""
    action: str
    confidence: float
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
device = None
transform = None


# FastAPI app
app = FastAPI(
    title="Action Recognition API",
    description="API for recognizing human actions in images using CNN-LSTM model with spatial attention",
    version="1.0.0"
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
    """Load model and action labels on startup."""
    global model, action_labels, device, transform

    print("="*80)
    print("Starting Action Recognition API Server")
    print("="*80)

    # Paths - relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "checkpoints", "actions", "best_model.pth")
    labels_path = os.path.join(base_dir, "checkpoints", "action_labels.pkl")

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Warning: Model checkpoint not found at {model_path}")
        print("API will start but action prediction will not work until model is available.")
        print(f"Please train the model first using: python run_action_training.py")
        return

    if not os.path.exists(labels_path):
        print(f"Warning: Action labels not found at {labels_path}")
        print("API will start but action prediction will not work until labels are available.")
        print(f"Please build labels first using: python scripts/build_action_labels.py")
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
    print(f"Loaded {action_labels.num_classes} action classes")

    # Load model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Get num_classes from checkpoint
    num_classes = checkpoint.get('num_classes', 40)

    # Create model
    model = create_action_model(num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Print model info
    if 'epoch' in checkpoint:
        print(f"Model trained for {checkpoint['epoch']} epochs")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")

    # Get inference transform
    transform = get_inference_transform()

    print("\n" + "="*80)
    print("API Server Ready!")
    print("="*80)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Action Recognition API",
        "version": "1.0.0",
        "model": "CNN-LSTM with Spatial Attention",
        "dataset": "Stanford 40 Actions",
        "endpoints": {
            "/api/predict": "POST - Predict action for an image",
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
    Predict action for an uploaded image.

    Args:
        file: Uploaded image file
        top_k: Number of top predictions to return (1-10)
        include_attention: Whether to include attention heatmap in response

    Returns:
        ActionResponse with predicted action and details
    """
    # Check if model and labels are loaded
    if model is None or action_labels is None:
        raise HTTPException(
            status_code=503,
            detail="Model or action labels not loaded. Please check server logs."
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

        # Predict action
        with torch.no_grad():
            logits, attention_weights = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, action_labels.num_classes))

        # Format top-k predictions
        top_k_predictions = []
        for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
            top_k_predictions.append(PredictionDetail(
                action=action_labels.decode_action(idx.item()),
                confidence=round(prob.item(), 4)
            ))

        processing_time = time.time() - start_time

        # Prepare attention heatmap if requested
        attention_heatmap = None
        if include_attention:
            attention_map = attention_weights[0].view(7, 7).cpu().numpy()
            attention_heatmap = attention_map.tolist()

        return ActionResponse(
            action=top_k_predictions[0].action,
            confidence=top_k_predictions[0].confidence,
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
