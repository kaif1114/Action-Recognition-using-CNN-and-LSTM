"""Comprehensive evaluation script for action recognition model."""
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import create_action_model
from data.action_labels import ActionLabels
from data.stanford40_dataset import Stanford40Dataset, stanford40_collate_fn
from data.transforms import get_transforms
from training.validate_actions import validate_with_confusion_matrix


def plot_confusion_matrix(conf_matrix, action_labels, save_path):
    """
    Plot and save confusion matrix.

    Args:
        conf_matrix: Confusion matrix (40x40)
        action_labels: ActionLabels object
        save_path: Path to save figure
    """
    plt.figure(figsize=(20, 18))

    # Get action names
    action_names = [action_labels.decode_action(i) for i in range(action_labels.num_classes)]
    action_names = [name.replace('_', ' ').title() for name in action_names]

    # Plot confusion matrix
    sns.heatmap(
        conf_matrix,
        annot=False,
        fmt='d',
        cmap='Blues',
        xticklabels=action_names,
        yticklabels=action_names,
        cbar_kws={'label': 'Number of Predictions'}
    )

    plt.xlabel('Predicted Action', fontsize=14, fontweight='bold')
    plt.ylabel('True Action', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix - Action Recognition', fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_per_class_accuracy(metrics, action_labels, save_path):
    """
    Plot per-class accuracy bar chart.

    Args:
        metrics: Validation metrics dictionary
        action_labels: ActionLabels object
        save_path: Path to save figure
    """
    per_class_acc = metrics['per_class_acc']

    # Sort by accuracy
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    class_indices = [idx for idx, _ in sorted_classes]
    accuracies = [acc for _, acc in sorted_classes]

    # Get action names
    action_names = [action_labels.decode_action(idx).replace('_', ' ').title() for idx in class_indices]

    # Create figure
    plt.figure(figsize=(14, 12))

    # Color bars based on accuracy
    colors = ['#2ecc71' if acc >= 0.7 else '#f39c12' if acc >= 0.5 else '#e74c3c' for acc in accuracies]

    # Plot bars
    plt.barh(range(len(action_names)), accuracies, color=colors)
    plt.yticks(range(len(action_names)), action_names)
    plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlim(0, 1)

    # Add accuracy values on bars
    for i, acc in enumerate(accuracies):
        plt.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Good (≥70%)'),
        Patch(facecolor='#f39c12', label='Fair (50-70%)'),
        Patch(facecolor='#e74c3c', label='Poor (<50%)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Per-class accuracy plot saved to: {save_path}")
    plt.close()


def plot_top_mistakes(conf_matrix, action_labels, save_path, top_n=10):
    """
    Plot top confusion pairs (most common mistakes).

    Args:
        conf_matrix: Confusion matrix
        action_labels: ActionLabels object
        save_path: Path to save figure
        top_n: Number of top mistakes to show
    """
    # Find top confusions (excluding diagonal)
    mistakes = []
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if i != j and conf_matrix[i, j] > 0:
                mistakes.append((i, j, conf_matrix[i, j]))

    # Sort by frequency
    mistakes.sort(key=lambda x: x[2], reverse=True)
    top_mistakes = mistakes[:top_n]

    # Prepare data
    labels = []
    counts = []
    for true_idx, pred_idx, count in top_mistakes:
        true_action = action_labels.decode_action(true_idx).replace('_', ' ').title()
        pred_action = action_labels.decode_action(pred_idx).replace('_', ' ').title()
        labels.append(f"{true_action}\n→ {pred_action}")
        counts.append(count)

    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(labels)))
    plt.barh(range(len(labels)), counts, color=colors)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Number of Misclassifications', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Confusion Pairs', fontsize=14, fontweight='bold')

    # Add count values
    for i, count in enumerate(counts):
        plt.text(count + 0.5, i, str(int(count)), va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Top mistakes plot saved to: {save_path}")
    plt.close()


def main():
    """Main evaluation function."""
    print("=" * 80)
    print("Action Recognition Model Evaluation")
    print("=" * 80)

    # Configuration
    config = {
        'checkpoint_path': 'checkpoints/actions/best_model.pth',
        'labels_path': 'checkpoints/action_labels.pkl',
        'image_dir': 'dataset/JPEGImages',
        'output_dir': 'outputs/evaluation',
        'batch_size': 32,
        'num_workers': 4
    }

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Check if checkpoint exists
    if not os.path.exists(config['checkpoint_path']):
        print(f"Error: Checkpoint not found at {config['checkpoint_path']}")
        print("Please train the model first using: python run_action_training.py")
        return

    # Load action labels
    print(f"\nLoading action labels from {config['labels_path']}...")
    action_labels = ActionLabels.load(config['labels_path'])
    print(f"Loaded {action_labels.num_classes} action classes")

    # Load model
    print(f"\nLoading model from {config['checkpoint_path']}...")
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    num_classes = checkpoint.get('num_classes', 40)

    model = create_action_model(num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    if 'val_acc' in checkpoint:
        print(f"  Best validation accuracy: {checkpoint['val_acc']:.4f}")

    # Get transforms
    _, val_transform = get_transforms()

    # Create test dataset
    print(f"\nLoading test dataset...")
    test_dataset = Stanford40Dataset(
        image_dir=config['image_dir'],
        split_file='test',
        action_labels=action_labels,
        transform=val_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=stanford40_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Test dataset: {len(test_dataset)} images, {len(test_loader)} batches")

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Running Evaluation on Test Set")
    print("=" * 80)

    metrics = validate_with_confusion_matrix(model, test_loader, criterion, device, action_labels)

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"\nOverall Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")

    # Save metrics to JSON
    metrics_path = os.path.join(config['output_dir'], 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        metrics_json = {
            'loss': float(metrics['loss']),
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'top5_accuracy': float(metrics['top5_accuracy'])
        }
        json.dump(metrics_json, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save classification report
    report_path = os.path.join(config['output_dir'], 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Action Recognition - Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(metrics['classification_report'])
    print(f"Classification report saved to: {report_path}")

    # Plot confusion matrix
    print("\nGenerating visualizations...")
    conf_matrix_path = os.path.join(config['output_dir'], 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], action_labels, conf_matrix_path)

    # Plot per-class accuracy (using metrics dict)
    per_class_acc_path = os.path.join(config['output_dir'], 'per_class_accuracy.png')

    # Calculate per-class accuracy from confusion matrix
    per_class_acc = {}
    for i in range(action_labels.num_classes):
        class_total = metrics['confusion_matrix'][i, :].sum()
        class_correct = metrics['confusion_matrix'][i, i]
        per_class_acc[i] = class_correct / class_total if class_total > 0 else 0.0

    metrics['per_class_acc'] = per_class_acc
    plot_per_class_accuracy(metrics, action_labels, per_class_acc_path)

    # Plot top mistakes
    top_mistakes_path = os.path.join(config['output_dir'], 'top_mistakes.png')
    plot_top_mistakes(metrics['confusion_matrix'], action_labels, top_mistakes_path, top_n=10)

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print(f"Results saved to: {config['output_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
