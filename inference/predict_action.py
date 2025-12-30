"""Action prediction and inference utilities."""
import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from models.model import create_action_model
from data.action_labels import ActionLabels
from data.transforms import get_inference_transform


def load_action_model(checkpoint_path, device='cuda'):
    """
    Load trained action recognition model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded ActionRecognitionModel
        action_labels: ActionLabels object
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get num_classes from checkpoint
    num_classes = checkpoint.get('num_classes', 40)

    # Create model
    model = create_action_model(num_classes=num_classes, device=device)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load action labels
    labels_path = 'checkpoints/action_labels.pkl'
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Action labels not found at {labels_path}")
    action_labels = ActionLabels.load(labels_path)

    print(f"Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")

    return model, action_labels


def predict_action(image_path, model, action_labels, device='cuda', top_k=5):
    """
    Predict action for a single image.

    Args:
        image_path: Path to image file
        model: ActionRecognitionModel
        action_labels: ActionLabels object
        device: Device
        top_k: Number of top predictions to return

    Returns:
        Dictionary with prediction results
    """
    # Load and transform image
    transform = get_inference_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        result = model.predict(image_tensor, action_labels, device, top_k=top_k)

    # Add image path to result
    result['image_path'] = image_path
    result['image_size'] = image.size

    return result


def visualize_attention(image_path, attention_map, save_path=None, alpha=0.4):
    """
    Visualize attention heatmap overlaid on image.

    Args:
        image_path: Path to original image
        attention_map: Attention weights (7, 7) numpy array
        save_path: Path to save visualization (optional)
        alpha: Transparency for heatmap overlay

    Returns:
        overlay: PIL Image with attention visualization
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Resize attention map to image size
    attention_resized = cv2.resize(attention_map, (image.width, image.height))

    # Normalize attention to 0-255
    attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
    attention_normalized = (attention_normalized * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(attention_normalized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    overlay_pil = Image.fromarray(overlay)

    # Save if requested
    if save_path:
        overlay_pil.save(save_path)
        print(f"Attention visualization saved to {save_path}")

    return overlay_pil


def visualize_prediction(image_path, result, save_path=None):
    """
    Create comprehensive prediction visualization.

    Args:
        image_path: Path to image
        result: Prediction result dictionary
        save_path: Path to save figure

    Returns:
        Figure object
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Original image
    image = Image.open(image_path).convert('RGB')
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 2. Attention heatmap
    attention_overlay = visualize_attention(image_path, result['attention_map'])
    axes[1].imshow(attention_overlay)
    axes[1].set_title('Attention Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 3. Top-K predictions bar chart
    top_k_actions = [pred['action'].replace('_', ' ').title() for pred in result['top_k']]
    top_k_confidences = [pred['confidence'] for pred in result['top_k']]

    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top_k_actions))]
    axes[2].barh(range(len(top_k_actions)), top_k_confidences, color=colors)
    axes[2].set_yticks(range(len(top_k_actions)))
    axes[2].set_yticklabels(top_k_actions)
    axes[2].set_xlabel('Confidence', fontsize=12)
    axes[2].set_title('Top-5 Predictions', fontsize=14, fontweight='bold')
    axes[2].set_xlim(0, 1)
    axes[2].invert_yaxis()

    # Add confidence values on bars
    for i, conf in enumerate(top_k_confidences):
        axes[2].text(conf + 0.01, i, f'{conf:.3f}', va='center', fontsize=10)

    # Add main prediction as suptitle
    predicted_action = result['action'].replace('_', ' ').title()
    fig.suptitle(f'Predicted Action: {predicted_action} (Confidence: {result["confidence"]:.3f})',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")

    return fig


def batch_predict(image_dir, model, action_labels, device='cuda', output_dir='outputs'):
    """
    Predict actions for all images in a directory.

    Args:
        image_dir: Directory containing images
        model: ActionRecognitionModel
        action_labels: ActionLabels object
        device: Device
        output_dir: Directory to save results

    Returns:
        List of prediction results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_dir)
                   if any(f.lower().endswith(ext) for ext in image_extensions)]

    print(f"Found {len(image_files)} images in {image_dir}")

    results = []
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        print(f"\n[{i}/{len(image_files)}] Processing {image_file}...")

        try:
            # Predict
            result = predict_action(image_path, model, action_labels, device)
            results.append(result)

            # Print prediction
            print(f"  Predicted: {result['action']} (confidence: {result['confidence']:.4f})")
            print(f"  Top-5: {[p['action'] for p in result['top_k']]}")

            # Save visualization
            vis_save_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_prediction.png")
            visualize_prediction(image_path, result, vis_save_path)

        except Exception as e:
            print(f"  Error processing {image_file}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Batch prediction completed! Processed {len(results)} images.")
    print(f"Results saved to {output_dir}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Action Recognition Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/actions/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                        help='Output directory for visualizations')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("\nLoading model...")
    model, action_labels = load_action_model(args.checkpoint, device)

    if args.image:
        # Single image prediction
        print(f"\nPredicting action for: {args.image}")
        result = predict_action(args.image, model, action_labels, device, top_k=args.top_k)

        # Print results
        print(f"\n{'=' * 60}")
        print("Prediction Results")
        print("=" * 60)
        print(f"Image: {result['image_path']}")
        print(f"Image size: {result['image_size']}")
        print(f"\nPredicted action: {result['action']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nTop-{args.top_k} predictions:")
        for i, pred in enumerate(result['top_k'], 1):
            print(f"  {i}. {pred['action']}: {pred['confidence']:.4f}")

        # Create visualization
        os.makedirs(args.output_dir, exist_ok=True)
        vis_path = os.path.join(args.output_dir, 'prediction_visualization.png')
        visualize_prediction(args.image, result, vis_path)

        # Save attention heatmap
        attention_path = os.path.join(args.output_dir, 'attention_heatmap.png')
        visualize_attention(args.image, result['attention_map'], attention_path)

        print(f"\n{'=' * 60}")
        print(f"Visualizations saved to {args.output_dir}")
        print("=" * 60)

    elif args.image_dir:
        # Batch prediction
        print(f"\nProcessing images in: {args.image_dir}")
        results = batch_predict(args.image_dir, model, action_labels, device, args.output_dir)

    else:
        print("Error: Please specify --image or --image_dir")
        parser.print_help()
