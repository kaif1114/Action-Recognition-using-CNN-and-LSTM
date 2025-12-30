"""Main training script for action recognition."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.model import create_action_model
from data.action_labels import ActionLabels
from data.stanford40_dataset import Stanford40Dataset, stanford40_collate_fn
from data.transforms import get_transforms
from training.train import train_model_classification


def main():
    """Main training function."""
    print("=" * 60)
    print("Action Recognition Training")
    print("Stanford 40 Actions Dataset")
    print("=" * 60)

    # Configuration
    config = {
        # Dataset paths
        'image_dir': 'dataset/JPEGImages',
        'splits_dir': 'dataset/ImageSplits',
        'label_path': 'checkpoints/action_labels.pkl',
        'checkpoint_dir': 'checkpoints/actions',

        # Training hyperparameters
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate_encoder': 1e-4,
        'learning_rate_decoder': 1e-3,
        'weight_decay': 1e-5,

        # Model hyperparameters
        'encoder_dim': 512,
        'decoder_dim': 512,
        'lstm_steps': 4,
        'num_classes': 40,
        'dropout': 0.5,
        'fine_tune_encoder': True,

        # Data split
        'val_split': 0.2,
        'random_seed': 42,

        # Training settings
        'num_workers': 4,
        'print_freq': 50,
        'save_freq': 5,

        # Scheduler
        'scheduler_factor': 0.5,
        'scheduler_patience': 5,
        'scheduler_min_lr': 1e-7,
    }

    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(config['random_seed'])

    # Load action labels
    print(f"\nLoading action labels from {config['label_path']}...")
    if not os.path.exists(config['label_path']):
        print(f"Error: Action labels not found at {config['label_path']}")
        print("Please run scripts/build_action_labels.py first.")
        return
    action_labels = ActionLabels.load(config['label_path'])
    print(f"Loaded {action_labels.num_classes} action classes")

    # Get transforms
    print("\nSetting up data transforms...")
    train_transform, val_transform = get_transforms()

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = Stanford40Dataset(
        image_dir=config['image_dir'],
        split_file='train',
        action_labels=action_labels,
        transform=train_transform,
        is_validation=False,
        val_split=config['val_split'],
        random_seed=config['random_seed']
    )

    val_dataset = Stanford40Dataset(
        image_dir=config['image_dir'],
        split_file='train',
        action_labels=action_labels,
        transform=val_transform,
        is_validation=True,
        val_split=config['val_split'],
        random_seed=config['random_seed']
    )

    test_dataset = Stanford40Dataset(
        image_dir=config['image_dir'],
        split_file='test',
        action_labels=action_labels,
        transform=val_transform
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=stanford40_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=stanford40_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_action_model(
        num_classes=config['num_classes'],
        encoder_dim=config['encoder_dim'],
        decoder_dim=config['decoder_dim'],
        lstm_steps=config['lstm_steps'],
        dropout=config['dropout'],
        fine_tune=config['fine_tune_encoder'],
        device=device
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create loss criterion
    criterion = nn.CrossEntropyLoss()

    # Create optimizer with different learning rates for encoder and decoder
    print("\nSetting up optimizer...")
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': config['learning_rate_encoder']},
        {'params': decoder_params, 'lr': config['learning_rate_decoder']}
    ], weight_decay=config['weight_decay'])

    print(f"  Encoder LR: {config['learning_rate_encoder']}")
    print(f"  Decoder LR: {config['learning_rate_decoder']}")
    print(f"  Weight decay: {config['weight_decay']}")

    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        min_lr=config['scheduler_min_lr'],
        verbose=True
    )

    print(f"\nScheduler: ReduceLROnPlateau")
    print(f"  Factor: {config['scheduler_factor']}")
    print(f"  Patience: {config['scheduler_patience']}")
    print(f"  Min LR: {config['scheduler_min_lr']}")

    # Check for existing checkpoint to resume training
    checkpoint_dir = config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_val_acc = 0.0

    resume_checkpoint = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(resume_checkpoint):
        print(f"\nFound existing checkpoint: {resume_checkpoint}")
        response = input("Resume training from checkpoint? (y/n): ")
        if response.lower() == 'y':
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('val_acc', 0.0)
            print(f"Resumed from epoch {start_epoch} (Best acc: {best_val_acc:.4f})")

    # Train model
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    trained_model = train_model_classification(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config['num_epochs'],
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
        print_freq=config['print_freq']
    )

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=stanford40_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    from training.validate_actions import validate_classification
    test_metrics = validate_classification(trained_model, test_loader, criterion, device)

    print(f"\nTest Set Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Top-5 Accuracy: {test_metrics['top5_accuracy']:.4f}")

    # Save test results
    import json
    results_path = os.path.join(checkpoint_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        test_results = {
            'loss': float(test_metrics['loss']),
            'accuracy': float(test_metrics['accuracy']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'top5_accuracy': float(test_metrics['top5_accuracy'])
        }
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
