"""Validation functions for action recognition model."""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def validate_classification(model, val_loader, criterion, device):
    """
    Validate classification model.

    Args:
        model: ActionRecognitionModel
        val_loader: Validation data loader
        criterion: Loss criterion (CrossEntropyLoss)
        device: Device to run on

    Returns:
        Dictionary with validation metrics:
            - loss: Average validation loss
            - accuracy: Overall accuracy
            - precision: Weighted precision
            - recall: Weighted recall
            - f1: Weighted F1 score
            - top5_accuracy: Top-5 accuracy
            - per_class_acc: Per-class accuracy dictionary
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    losses = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits, attention_weights = model(images)
            loss = criterion(logits, labels)

            # Get probabilities and predictions
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            losses.append(loss.item())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    # Top-5 accuracy
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_acc = np.mean([label in top5_preds[i] for i, label in enumerate(all_labels)])

    # Per-class accuracy
    per_class_acc = {}
    for class_idx in range(40):
        class_mask = all_labels == class_idx
        if class_mask.sum() > 0:
            class_correct = (all_preds[class_mask] == class_idx).sum()
            per_class_acc[class_idx] = class_correct / class_mask.sum()
        else:
            per_class_acc[class_idx] = 0.0

    return {
        'loss': np.mean(losses),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top5_accuracy': top5_acc,
        'per_class_acc': per_class_acc
    }


def validate_with_confusion_matrix(model, val_loader, criterion, device, action_labels):
    """
    Validate model and return confusion matrix.

    Args:
        model: ActionRecognitionModel
        val_loader: Validation data loader
        criterion: Loss criterion
        device: Device
        action_labels: ActionLabels object

    Returns:
        Dictionary with metrics and confusion matrix
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    losses = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images)
            loss = criterion(logits, labels)

            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            losses.append(loss.item())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate all metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    # Top-5 accuracy
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_acc = np.mean([label in top5_preds[i] for i, label in enumerate(all_labels)])

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Classification report
    target_names = [action_labels.decode_action(i) for i in range(action_labels.num_classes)]
    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        zero_division=0
    )

    return {
        'loss': np.mean(losses),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top5_accuracy': top5_acc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }


def get_top_k_predictions(model, image, action_labels, device, k=5):
    """
    Get top-k predictions for a single image.

    Args:
        model: ActionRecognitionModel
        image: Image tensor (3, 224, 224) or (1, 3, 224, 224)
        action_labels: ActionLabels object
        device: Device
        k: Number of top predictions

    Returns:
        List of tuples (action_name, confidence)
    """
    model.eval()

    # Ensure batch dimension
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        logits, _ = model(image)
        probs = F.softmax(logits, dim=1)
        top_k_probs, top_k_indices = torch.topk(probs, k)

    predictions = []
    for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
        action = action_labels.decode_action(idx.item())
        predictions.append((action, prob.item()))

    return predictions


if __name__ == "__main__":
    # Test validation functions
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models.model import create_action_model
    from data.action_labels import ActionLabels
    from data.stanford40_dataset import Stanford40Dataset, stanford40_collate_fn
    from data.transforms import get_transforms
    from torch.utils.data import DataLoader
    import torch.nn as nn

    print("=" * 60)
    print("Testing Validation Functions")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load action labels
    action_labels = ActionLabels.load('checkpoints/action_labels.pkl')

    # Create model
    print("\nCreating model...")
    model = create_action_model(num_classes=40, device=device)

    # Get transforms
    _, val_transform = get_transforms()

    # Create small validation dataset
    print("\nLoading validation dataset...")
    val_dataset = Stanford40Dataset(
        image_dir="dataset/JPEGImages",
        split_file='train',
        action_labels=action_labels,
        transform=val_transform,
        is_validation=True,
        val_split=0.2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=stanford40_collate_fn,
        num_workers=0
    )

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Test validation
    print("\nRunning validation...")
    metrics = validate_classification(model, val_loader, criterion, device)

    print(f"\nValidation Results:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")

    print(f"\n{'=' * 60}")
    print("Validation test completed successfully!")
    print("=" * 60)
