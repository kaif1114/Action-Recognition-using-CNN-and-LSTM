"""Training script for image caption generation."""
import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.vocabulary import Vocabulary
from data.coco_caption_dataset import COCOCaptionDataset, coco_caption_collate_fn
from data.transforms import get_transforms
from models.encoder import Encoder
from models.caption_decoder import DecoderWithAttention
from training.validate_captions import validate_caption_model


def load_pretrained_encoder(checkpoint_path, device):
    """
    Load pre-trained encoder from action recognition checkpoint.

    Args:
        checkpoint_path: Path to action model checkpoint
        device: Device to load on

    Returns:
        encoder: Loaded and frozen encoder
    """
    print(f"\nLoading pre-trained encoder from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create encoder
    encoder = Encoder(encoded_size=512, fine_tune=False)

    # Extract encoder state dict from model state dict
    # The checkpoint contains the full ActionRecognitionModel
    model_state = checkpoint['model_state_dict']

    # Filter encoder weights (keys starting with 'encoder.')
    encoder_state = {
        k.replace('encoder.', ''): v
        for k, v in model_state.items()
        if k.startswith('encoder.')
    }

    # Load encoder weights
    encoder.load_state_dict(encoder_state)

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()  # Set to eval mode
    encoder.to(device)

    print(f"✓ Encoder loaded and frozen")
    print(f"✓ Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    return encoder


def train_epoch(encoder, decoder, train_loader, criterion, optimizer, device, epoch, grad_clip=5.0):
    """
    Train for one epoch.

    Args:
        encoder: Frozen encoder
        decoder: Caption decoder
        train_loader: Training data loader
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number
        grad_clip: Gradient clipping value

    Returns:
        Average training loss
    """
    decoder.train()
    encoder.eval()  # Encoder stays in eval mode

    total_loss = 0
    num_batches = 0

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for images, captions, lengths in pbar:
        images = images.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)

        # Forward pass
        with torch.no_grad():
            encoder_out = encoder(images)  # Encoder is frozen

        predictions, alphas, sorted_captions, decode_lengths, sort_ind = decoder(
            encoder_out, captions, lengths
        )

        # Compute loss
        # Target: exclude <START> token (first token)
        targets = sorted_captions[:, 1:]  # Remove <START>

        # Pack predictions and targets
        batch_loss = 0
        for i in range(len(decode_lengths)):
            decode_len = decode_lengths[i]
            batch_loss += criterion(
                predictions[i, :decode_len, :],
                targets[i, :decode_len]
            )

        batch_loss = batch_loss / len(decode_lengths)  # Average over batch

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

        optimizer.step()

        # Update metrics
        total_loss += batch_loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def main():
    """Main training function."""
    print("=" * 80)
    print("Image Caption Training - COCO Actions Dataset")
    print("=" * 80)

    # Configuration
    config = {
        # Paths
        'image_dir': 'dataset/coco_actions/images',
        'captions_json': 'dataset/coco_actions/action_captions.json',
        'vocabulary_path': 'checkpoints/vocabulary.pkl',
        'action_checkpoint': 'checkpoints/actions/best_model.pth',
        'checkpoint_dir': 'checkpoints/captions',

        # Model
        'encoder_dim': 512,
        'attention_dim': 512,
        'embed_dim': 512,
        'decoder_dim': 512,
        'dropout': 0.5,
        'max_caption_len': 20,

        # Training
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-3,  # Higher LR since only decoder trains
        'weight_decay': 1e-5,
        'gradient_clip': 5.0,
        'train_split': 0.8,
        'num_workers': 0,  # Windows compatible

        # Validation
        'val_every': 1,  # Validate every N epochs
        'save_every': 1,  # Save checkpoint every N epochs
        'num_val_samples': 5,  # Number of sample captions to generate

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Print configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    device = torch.device(config['device'])
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Load vocabulary
    print("\n" + "=" * 80)
    print("Loading Vocabulary")
    print("=" * 80)

    if not os.path.exists(config['vocabulary_path']):
        print(f"Error: Vocabulary not found at {config['vocabulary_path']}")
        print("Please run: python scripts/build_vocabulary.py")
        return

    vocabulary = Vocabulary.load(config['vocabulary_path'])
    config['vocab_size'] = len(vocabulary)
    print(f"Vocabulary size: {config['vocab_size']}")

    # Create datasets
    print("\n" + "=" * 80)
    print("Creating Datasets")
    print("=" * 80)

    train_transform, val_transform = get_transforms()

    train_dataset = COCOCaptionDataset(
        image_dir=config['image_dir'],
        captions_json=config['captions_json'],
        vocabulary=vocabulary,
        transform=train_transform,
        split='train',
        train_split=config['train_split'],
        max_caption_len=config['max_caption_len']
    )

    val_dataset = COCOCaptionDataset(
        image_dir=config['image_dir'],
        captions_json=config['captions_json'],
        vocabulary=vocabulary,
        transform=val_transform,
        split='val',
        train_split=config['train_split'],
        max_caption_len=config['max_caption_len']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=coco_caption_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=coco_caption_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Load pre-trained encoder
    print("\n" + "=" * 80)
    print("Loading Pre-trained Encoder")
    print("=" * 80)

    if not os.path.exists(config['action_checkpoint']):
        print(f"Error: Action checkpoint not found at {config['action_checkpoint']}")
        print("Please train action model first: python run_action_training.py")
        return

    encoder = load_pretrained_encoder(config['action_checkpoint'], device)

    # Create caption decoder
    print("\n" + "=" * 80)
    print("Creating Caption Decoder")
    print("=" * 80)

    decoder = DecoderWithAttention(
        attention_dim=config['attention_dim'],
        embed_dim=config['embed_dim'],
        decoder_dim=config['decoder_dim'],
        vocab_size=config['vocab_size'],
        encoder_dim=config['encoder_dim'],
        dropout=config['dropout']
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Decoder parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(decoder.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    print(f"\nOptimizer: Adam")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Gradient clipping: {config['gradient_clip']}")

    # Check for resume
    start_epoch = 1
    best_val_loss = float('inf')
    best_perplexity = float('inf')

    # Find latest checkpoint
    checkpoint_files = [f for f in os.listdir(config['checkpoint_dir'])
                       if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if checkpoint_files:
        # Extract epoch numbers and find the latest
        epoch_numbers = [int(f.replace('checkpoint_epoch_', '').replace('.pth', ''))
                        for f in checkpoint_files]
        latest_epoch = max(epoch_numbers)
        latest_checkpoint = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{latest_epoch}.pth')

        print("\n" + "=" * 80)
        print(f"Found checkpoint: checkpoint_epoch_{latest_epoch}.pth")
        print("=" * 80)

        resume = input(f"Resume training from epoch {latest_epoch}? (y/n): ").strip().lower()

        if resume == 'y':
            print(f"Loading checkpoint from epoch {latest_epoch}...")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)

            # Load model and optimizer states
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = latest_epoch + 1

            # Load best metrics if available
            if os.path.exists(os.path.join(config['checkpoint_dir'], 'best_model.pth')):
                best_checkpoint = torch.load(os.path.join(config['checkpoint_dir'], 'best_model.pth'),
                                            map_location=device, weights_only=False)
                best_val_loss = best_checkpoint.get('val_loss', float('inf'))
                best_perplexity = best_checkpoint.get('perplexity', float('inf'))
                print(f"Best validation perplexity so far: {best_perplexity:.2f}")

            print(f"✓ Resuming from epoch {start_epoch}")
            print(f"✓ Will train epochs {start_epoch} to {config['num_epochs']}")

    # Training loop
    print("\n" + "=" * 80)
    print(f"Starting Training - Epochs {start_epoch} to {config['num_epochs']}")
    print("=" * 80)

    for epoch in range(start_epoch, config['num_epochs'] + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print("=" * 80)

        # Train
        train_loss = train_epoch(
            encoder, decoder, train_loader, criterion, optimizer,
            device, epoch, config['gradient_clip']
        )

        train_perplexity = math.exp(min(train_loss, 100))

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Perplexity: {train_perplexity:.2f}")

        # Validate
        if epoch % config['val_every'] == 0:
            print(f"\nRunning validation...")
            val_results = validate_caption_model(
                encoder, decoder, val_loader, criterion, vocabulary,
                device, num_samples=config['num_val_samples']
            )

            val_loss = val_results['loss']
            val_perplexity = val_results['perplexity']

            print(f"\nValidation Loss: {val_loss:.4f}")
            print(f"Validation Perplexity: {val_perplexity:.2f}")

            # Print sample captions
            print(f"\nSample Captions:")
            for i, (gt, pred) in enumerate(zip(
                val_results['sample_captions_gt'][:config['num_val_samples']],
                val_results['sample_captions_pred'][:config['num_val_samples']]
            )):
                print(f"  [{i+1}] GT:   {gt}")
                print(f"      Pred: {pred}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_perplexity = val_perplexity

                best_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'perplexity': val_perplexity,
                    'vocab_size': config['vocab_size'],
                    'config': config
                }, best_path)
                print(f"\n✓ Best model saved (perplexity: {best_perplexity:.2f})")

        # Save periodic checkpoint
        if epoch % config['save_every'] == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_perplexity': train_perplexity,
                'vocab_size': config['vocab_size'],
                'config': config
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")

    # Training complete
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation perplexity: {best_perplexity:.2f}")
    print(f"Best model saved to: {os.path.join(config['checkpoint_dir'], 'best_model.pth')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
