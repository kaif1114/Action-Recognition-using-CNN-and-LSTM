"""Combined inference for action recognition and caption generation."""
import os
import sys
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vocabulary import Vocabulary
from data.action_labels import ActionLabels
from data.transforms import get_inference_transform
from models.unified_model import load_unified_model


def predict_combined(image_path, model, action_labels, vocabulary, device='cuda', top_k=5):
    """
    Predict both action and caption for an image.

    Args:
        image_path: Path to image file
        model: Unified model
        action_labels: ActionLabels object
        vocabulary: Vocabulary object
        device: Device
        top_k: Number of top action predictions

    Returns:
        Dictionary with combined results
    """
    # Load and transform image
    transform = get_inference_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    model.eval()
    result = model.predict_combined(image_tensor, vocabulary, action_labels, device, top_k=top_k)

    # Add image path and size
    result['image_path'] = image_path
    result['image_size'] = image.size

    return result


def visualize_attention(image_path, attention_map, save_path=None, alpha=0.4):
    """
    Visualize attention heatmap overlaid on image.

    Args:
        image_path: Path to original image
        attention_map: Attention weights (7, 7) numpy array
        save_path: Path to save visualization
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
    attention_normalized = (attention_resized - attention_resized.min()) / (
        attention_resized.max() - attention_resized.min() + 1e-8
    )
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


def visualize_combined(image_path, result, save_path=None):
    """
    Create comprehensive visualization with action and caption.

    Args:
        image_path: Path to image
        result: Prediction result dictionary
        save_path: Path to save figure

    Returns:
        Figure object
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Original image
    image = Image.open(image_path).convert('RGB')
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 2. Attention heatmap
    attention_overlay = visualize_attention(image_path, result['attention_map'])
    axes[1].imshow(attention_overlay)
    axes[1].set_title('Action Attention Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 3. Results panel
    axes[2].axis('off')

    # Caption at top
    caption_text = f'"{result["caption"]}"'
    axes[2].text(0.5, 0.85, 'Generated Caption:', ha='center', fontsize=14,
                 fontweight='bold', transform=axes[2].transAxes)
    axes[2].text(0.5, 0.75, caption_text, ha='center', va='top', fontsize=12,
                 style='italic', wrap=True, transform=axes[2].transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Top-K actions below
    axes[2].text(0.5, 0.6, 'Top-5 Actions:', ha='center', fontsize=14,
                 fontweight='bold', transform=axes[2].transAxes)

    y_pos = 0.5
    for i, pred in enumerate(result['top_k'][:5]):
        action_name = pred['action'].replace('_', ' ').title()
        confidence = pred['confidence']
        color = '#2ecc71' if i == 0 else '#3498db'

        # Action text
        axes[2].text(0.1, y_pos, f"{i+1}. {action_name}",
                    fontsize=11, transform=axes[2].transAxes)

        # Confidence bar
        bar_width = confidence * 0.4
        axes[2].barh(y_pos, bar_width, height=0.03, left=0.55,
                    color=color, transform=axes[2].transAxes)

        # Confidence text
        axes[2].text(0.55 + bar_width + 0.02, y_pos, f'{confidence:.1%}',
                    fontsize=10, va='center', transform=axes[2].transAxes)

        y_pos -= 0.08

    # Main title
    predicted_action = result['action'].replace('_', ' ').title()
    fig.suptitle(
        f'Action: {predicted_action} ({result["confidence"]:.1%})',
        fontsize=16, fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Combined visualization saved to {save_path}")

    return fig


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Combined Action Recognition and Caption Generation')
    parser.add_argument('--action_checkpoint', type=str, default='checkpoints/actions/best_model.pth',
                        help='Path to action model checkpoint')
    parser.add_argument('--caption_checkpoint', type=str, default='checkpoints/captions/best_model.pth',
                        help='Path to caption model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='outputs/combined_predictions',
                        help='Output directory for visualizations')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top action predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Check if checkpoints exist
    if not os.path.exists(args.action_checkpoint):
        print(f"Error: Action checkpoint not found at {args.action_checkpoint}")
        return

    if not os.path.exists(args.caption_checkpoint):
        print(f"Error: Caption checkpoint not found at {args.caption_checkpoint}")
        return

    # Load action labels
    action_labels_path = 'checkpoints/action_labels.pkl'
    if not os.path.exists(action_labels_path):
        print(f"Error: Action labels not found at {action_labels_path}")
        return

    print("\nLoading action labels...")
    action_labels = ActionLabels.load(action_labels_path)

    # Load vocabulary
    vocabulary_path = 'checkpoints/vocabulary.pkl'
    if not os.path.exists(vocabulary_path):
        print(f"Error: Vocabulary not found at {vocabulary_path}")
        return

    print("Loading vocabulary...")
    vocabulary = Vocabulary.load(vocabulary_path)

    # Load unified model
    print("\nLoading unified model...")
    model = load_unified_model(args.action_checkpoint, args.caption_checkpoint, device)

    # Predict
    print(f"\nPredicting for: {args.image}")
    result = predict_combined(args.image, model, action_labels, vocabulary, device, top_k=args.top_k)

    # Print results
    print("\n" + "=" * 80)
    print("COMBINED PREDICTION RESULTS")
    print("=" * 80)
    print(f"Image: {result['image_path']}")
    print(f"Image size: {result['image_size']}")

    print(f"\n{'─' * 80}")
    print("ACTION RECOGNITION:")
    print("─" * 80)
    print(f"Predicted action: {result['action']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nTop-{args.top_k} predictions:")
    for i, pred in enumerate(result['top_k'], 1):
        print(f"  {i}. {pred['action']}: {pred['confidence']:.2%}")

    print(f"\n{'─' * 80}")
    print("CAPTION GENERATION:")
    print("─" * 80)
    print(f'"{result["caption"]}"')

    # Create visualization
    os.makedirs(args.output_dir, exist_ok=True)

    # Save combined visualization
    vis_path = os.path.join(args.output_dir, 'combined_prediction.png')
    visualize_combined(args.image, result, vis_path)

    # Save attention heatmap
    attention_path = os.path.join(args.output_dir, 'attention_heatmap.png')
    visualize_attention(args.image, result['attention_map'], attention_path)

    print(f"\n{'=' * 80}")
    print(f"Visualizations saved to {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
