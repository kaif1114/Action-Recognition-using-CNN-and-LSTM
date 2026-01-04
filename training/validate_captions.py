"""Validation functions for image caption generation."""
import torch
import torch.nn as nn
import math


def validate_caption_model(encoder, decoder, val_loader, criterion, vocabulary, device='cuda', num_samples=5):
    """
    Validate caption model on validation set.

    Args:
        encoder: Encoder model (frozen)
        decoder: Caption decoder model
        val_loader: Validation data loader
        criterion: Loss criterion (CrossEntropyLoss)
        vocabulary: Vocabulary object for decoding
        device: Device to run on
        num_samples: Number of sample captions to generate

    Returns:
        Dictionary with validation metrics
    """
    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_words = 0
    num_batches = 0

    # Lists to store sample captions
    sample_images = []
    sample_captions_gt = []
    sample_captions_pred = []

    with torch.no_grad():
        for batch_idx, (images, captions, lengths) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)
            lengths = lengths.to(device)

            # Forward pass
            encoder_out = encoder(images)
            predictions, alphas, sorted_captions, decode_lengths, sort_ind = decoder(
                encoder_out, captions, lengths
            )

            # Compute loss
            # Target: exclude <START> token (first token)
            # predictions: exclude last timestep (we don't predict after <END>)
            targets = sorted_captions[:, 1:]  # Remove <START>

            # Pack predictions and targets for loss computation
            batch_loss = 0
            for i in range(len(decode_lengths)):
                decode_len = decode_lengths[i]
                batch_loss += criterion(
                    predictions[i, :decode_len, :],
                    targets[i, :decode_len]
                )
                total_words += decode_len

            total_loss += batch_loss.item()
            num_batches += 1

            # Generate sample captions for first batch
            if batch_idx == 0 and num_samples > 0:
                for i in range(min(num_samples, len(images))):
                    # Store image and ground truth
                    sample_images.append(images[i])
                    gt_caption = vocabulary.decode_caption(captions[i].cpu().tolist())
                    sample_captions_gt.append(gt_caption)

                    # Generate caption (greedy decoding)
                    pred_caption = _generate_caption_greedy(
                        encoder_out[i:i+1], decoder, vocabulary, device, max_length=20
                    )
                    sample_captions_pred.append(pred_caption)

    # Calculate average loss per word
    avg_loss = total_loss / total_words if total_words > 0 else 0

    # Calculate perplexity
    perplexity = math.exp(min(avg_loss, 100))  # Cap to prevent overflow

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_batches': num_batches,
        'sample_captions_gt': sample_captions_gt,
        'sample_captions_pred': sample_captions_pred
    }


def _generate_caption_greedy(encoder_out, decoder, vocabulary, device, max_length=20):
    """
    Generate caption using greedy decoding.

    Args:
        encoder_out: Encoder output (1, num_pixels, encoder_dim)
        decoder: Decoder model
        vocabulary: Vocabulary object
        device: Device
        max_length: Maximum caption length

    Returns:
        Generated caption string
    """
    decoder.eval()

    # Initialize LSTM state
    h, c = decoder.init_hidden_state(encoder_out)

    # Start with <START> token
    current_word = torch.LongTensor([vocabulary.word2idx['<START>']]).to(device)
    generated_words = []

    for _ in range(max_length):
        # Embed current word
        embeddings = decoder.embedding(current_word).unsqueeze(1)  # (1, 1, embed_dim)
        embeddings = embeddings.squeeze(1)  # (1, embed_dim)

        # Attention
        context, alpha = decoder.attention(encoder_out, h)

        # Gate context
        gate = torch.sigmoid(decoder.f_beta(h))
        gated_context = gate * context

        # LSTM step
        h, c = decoder.decode_step(
            torch.cat([embeddings, gated_context], dim=1),
            (h, c)
        )

        # Predict next word
        scores = decoder.fc(h)  # (1, vocab_size)
        predicted_idx = scores.argmax(dim=1).item()

        # Check for <END>
        if predicted_idx == vocabulary.word2idx['<END>']:
            break

        # Add word (skip special tokens in output)
        word = vocabulary.idx2word.get(predicted_idx, '<UNK>')
        if word not in ['<PAD>', '<START>', '<END>']:
            generated_words.append(word)

        # Update current word
        current_word = torch.LongTensor([predicted_idx]).to(device)

    return ' '.join(generated_words) if generated_words else ""


if __name__ == "__main__":
    """Test validation function."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models.encoder import Encoder
    from models.caption_decoder import DecoderWithAttention
    from data.vocabulary import Vocabulary
    from data.coco_caption_dataset import COCOCaptionDataset, coco_caption_collate_fn
    from data.transforms import get_transforms
    from torch.utils.data import DataLoader

    print("=" * 80)
    print("Testing Caption Validation Function")
    print("=" * 80)

    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load vocabulary
    vocabulary_path = "checkpoints/vocabulary.pkl"
    if not os.path.exists(vocabulary_path):
        print(f"Error: Vocabulary not found. Run: python scripts/build_vocabulary.py")
        exit(1)

    print("\nLoading vocabulary...")
    vocabulary = Vocabulary.load(vocabulary_path)

    # Create models
    print("\nCreating models...")
    encoder = Encoder(encoded_size=512, fine_tune=False).to(device)
    decoder = DecoderWithAttention(
        attention_dim=512,
        embed_dim=512,
        decoder_dim=512,
        vocab_size=len(vocabulary),
        encoder_dim=512,
        dropout=0.5
    ).to(device)

    encoder.eval()  # Encoder is frozen
    decoder.train()

    # Create validation dataset
    print("\nCreating validation dataset...")
    _, val_transform = get_transforms()
    val_dataset = COCOCaptionDataset(
        image_dir="dataset/coco_actions/images",
        captions_json="dataset/coco_actions/action_captions.json",
        vocabulary=vocabulary,
        transform=val_transform,
        split='val',
        train_split=0.8
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=coco_caption_collate_fn,
        num_workers=0
    )

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Run validation
    print("\nRunning validation (on first 10 batches)...")
    val_loader_subset = list(val_loader)[:10]  # Only 10 batches for testing

    results = validate_caption_model(
        encoder, decoder, val_loader_subset, criterion, vocabulary, device, num_samples=3
    )

    print(f"\n{'-' * 80}")
    print("Validation Results")
    print("-" * 80)
    print(f"Average loss: {results['loss']:.4f}")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Batches processed: {results['num_batches']}")

    print(f"\nSample captions:")
    for i, (gt, pred) in enumerate(zip(results['sample_captions_gt'], results['sample_captions_pred'])):
        print(f"\nSample {i+1}:")
        print(f"  Ground truth: {gt}")
        print(f"  Predicted:    {pred}")

    print(f"\n{'=' * 80}")
    print("Validation function test completed successfully!")
    print("=" * 80)
