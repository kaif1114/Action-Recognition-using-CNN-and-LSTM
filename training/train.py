"""Training script for image captioning model."""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, print_freq=100):
    """
    Train for one epoch.

    Args:
        model: EncoderDecoder model
        train_loader: Training data loader
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        print_freq: Print frequency

    Returns:
        avg_loss: Average loss for the epoch
        avg_perplexity: Average perplexity
    """
    model.train()

    batch_time = []
    data_time = []
    losses = []

    start = time.time()

    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for i, (images, captions, caption_lengths) in enumerate(pbar):
        data_time.append(time.time() - start)

        # Move to device
        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        # Forward pass
        predictions, alphas, encoded_captions, decode_lengths, sort_ind = model(
            images, captions, caption_lengths
        )

        # Pack predictions and targets for loss calculation
        # predictions: (batch_size, max_decode_length, vocab_size)
        # We need to remove timesteps that we didn't decode at
        # This is done by packing the sequences

        # Targets are captions without start token
        targets = encoded_captions[:, 1:]  # (batch_size, max_caption_length - 1)

        # Pack both predictions and targets
        # Remove predictions and targets for padding
        predictions_packed = pack_padded_sequence(predictions, decode_lengths, batch_first=True)[0]
        targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(predictions_packed, targets_packed)

        # Add doubly stochastic attention regularization (optional)
        # This encourages the model to attend to every part of the image
        if alphas is not None:
            alpha_regularization = 1.0 * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            loss += alpha_regularization

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Update weights
        optimizer.step()

        # Record loss
        losses.append(loss.item())

        # Record batch time
        batch_time.append(time.time() - start)
        start = time.time()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'perplexity': f'{np.exp(loss.item()):.2f}'
        })

        # Print status
        if i % print_freq == 0 and i > 0:
            print(f'\nBatch {i}/{len(train_loader)}:')
            print(f'  Loss: {loss.item():.4f}')
            print(f'  Perplexity: {np.exp(loss.item()):.2f}')
            print(f'  Batch time: {np.mean(batch_time[-print_freq:]):.3f}s')
            print(f'  Data time: {np.mean(data_time[-print_freq:]):.3f}s')

    # Calculate average loss and perplexity
    avg_loss = np.mean(losses)
    avg_perplexity = np.exp(avg_loss)

    return avg_loss, avg_perplexity


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    checkpoint_dir,
    start_epoch=0,
    best_val_loss=float('inf'),
    print_freq=100
):
    """
    Train the model.

    Args:
        model: EncoderDecoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss criterion
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs to train
        checkpoint_dir: Directory to save checkpoints
        start_epoch: Starting epoch (for resuming training)
        best_val_loss: Best validation loss so far
        print_freq: Print frequency

    Returns:
        model: Trained model
    """
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training history
    train_losses = []
    val_losses = []

    # Early stopping
    patience = 5
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        print(f'\n{"="*60}')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'{"="*60}')

        # Train for one epoch
        train_loss, train_perplexity = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1, print_freq
        )

        print(f'\nTraining Summary:')
        print(f'  Average Loss: {train_loss:.4f}')
        print(f'  Average Perplexity: {train_perplexity:.2f}')

        # Validate
        from training.validate import validate
        val_loss, val_perplexity, bleu_scores = validate(model, val_loader, criterion, device)

        print(f'\nValidation Summary:')
        print(f'  Average Loss: {val_loss:.4f}')
        print(f'  Average Perplexity: {val_perplexity:.2f}')
        if bleu_scores:
            print(f'  BLEU-1: {bleu_scores.get("bleu1", 0):.4f}')
            print(f'  BLEU-2: {bleu_scores.get("bleu2", 0):.4f}')
            print(f'  BLEU-3: {bleu_scores.get("bleu3", 0):.4f}')
            print(f'  BLEU-4: {bleu_scores.get("bleu4", 0):.4f}')

        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  Learning Rate: {current_lr:.6f}')

        # Save training history
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'bleu_scores': bleu_scores,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'vocab_size': model.decoder.vocab_size
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'\nCheckpoint saved: {checkpoint_path}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f'Best model saved: {best_checkpoint_path}')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{patience}')

        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break

        # Print epoch time
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch time: {epoch_time / 60:.2f} minutes')

    print(f'\n{"="*60}')
    print('Training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'{"="*60}')

    return model


def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss, checkpoint_dir, is_best=False):
    """
    Save checkpoint.

    Args:
        epoch: Current epoch
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        train_loss: Training loss
        val_loss: Validation loss
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'vocab_size': model.decoder.vocab_size
    }

    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)

    Returns:
        epoch: Starting epoch
        train_loss: Training loss
        val_loss: Validation loss
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    train_loss = checkpoint.get('train_loss', 0.0)
    val_loss = checkpoint.get('val_loss', 0.0)

    print(f'Checkpoint loaded from {checkpoint_path}')
    print(f'  Epoch: {epoch}')
    print(f'  Train loss: {train_loss:.4f}')
    print(f'  Val loss: {val_loss:.4f}')

    return epoch, train_loss, val_loss


def train_epoch_classification(model, train_loader, criterion, optimizer, device, epoch, print_freq=100):
    """
    Train for one epoch (classification version for action recognition).

    Args:
        model: ActionRecognitionModel
        train_loader: Training data loader
        criterion: Loss criterion (CrossEntropyLoss)
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        print_freq: Print frequency

    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Classification accuracy
    """
    model.train()

    batch_time = []
    data_time = []
    losses = []
    correct = 0
    total = 0

    start = time.time()

    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for i, (images, labels) in enumerate(pbar):
        data_time.append(time.time() - start)

        # Move to device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits, attention_weights = model(images)
        # logits: (batch_size, num_classes)
        # attention_weights: (batch_size, 49)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Update weights
        optimizer.step()

        # Record loss
        losses.append(loss.item())

        # Calculate accuracy
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Record batch time
        batch_time.append(time.time() - start)
        start = time.time()

        # Update progress bar
        current_acc = correct / total if total > 0 else 0
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.4f}'
        })

        # Print status
        if i % print_freq == 0 and i > 0:
            print(f'\nBatch {i}/{len(train_loader)}:')
            print(f'  Loss: {loss.item():.4f}')
            print(f'  Accuracy: {current_acc:.4f}')
            print(f'  Batch time: {np.mean(batch_time[-print_freq:]):.3f}s')
            print(f'  Data time: {np.mean(data_time[-print_freq:]):.3f}s')

    # Calculate average loss and accuracy
    avg_loss = np.mean(losses)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


def train_model_classification(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    checkpoint_dir,
    start_epoch=0,
    best_val_acc=0.0,
    print_freq=100
):
    """
    Train the classification model.

    Args:
        model: ActionRecognitionModel
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss criterion (CrossEntropyLoss)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs to train
        checkpoint_dir: Directory to save checkpoints
        start_epoch: Starting epoch (for resuming training)
        best_val_acc: Best validation accuracy so far
        print_freq: Print frequency

    Returns:
        model: Trained model
    """
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # Early stopping
    patience = 10
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        print(f'\n{"="*60}')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'{"="*60}')

        # Train for one epoch
        train_loss, train_acc = train_epoch_classification(
            model, train_loader, criterion, optimizer, device, epoch + 1, print_freq
        )

        print(f'\nTraining Summary:')
        print(f'  Average Loss: {train_loss:.4f}')
        print(f'  Accuracy: {train_acc:.4f}')

        # Validate
        from training.validate_actions import validate_classification
        val_metrics = validate_classification(model, val_loader, criterion, device)

        print(f'\nValidation Summary:')
        print(f'  Average Loss: {val_metrics["loss"]:.4f}')
        print(f'  Accuracy: {val_metrics["accuracy"]:.4f}')
        print(f'  Precision: {val_metrics["precision"]:.4f}')
        print(f'  Recall: {val_metrics["recall"]:.4f}')
        print(f'  F1 Score: {val_metrics["f1"]:.4f}')
        print(f'  Top-5 Accuracy: {val_metrics["top5_accuracy"]:.4f}')

        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # For ReduceLROnPlateau, use validation loss
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  Learning Rate: {current_lr:.6f}')

        # Save training history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_metrics["loss"])
        val_accs.append(val_metrics["accuracy"])

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics["loss"],
            'val_acc': val_metrics["accuracy"],
            'val_metrics': val_metrics,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'num_classes': 40
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'\nCheckpoint saved: {checkpoint_path}')

        # Save best model based on validation accuracy
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f'Best model saved: {best_checkpoint_path} (Acc: {best_val_acc:.4f})')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{patience}')

        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break

        # Print epoch time
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch time: {epoch_time / 60:.2f} minutes')

    print(f'\n{"="*60}')
    print('Training completed!')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'{"="*60}')

    return model
