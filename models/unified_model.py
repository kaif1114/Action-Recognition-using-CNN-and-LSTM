"""Unified multi-task model for action recognition and caption generation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedActionCaptionModel(nn.Module):
    """
    Unified model combining action recognition and image captioning.

    Architecture:
        - Shared ResNet101 encoder (frozen after action training)
        - Action decoder: LSTM with 4 steps → 40 class logits
        - Caption decoder: LSTM with variable steps → caption tokens

    During inference, both decoders run on the same encoder output.
    """

    def __init__(self, encoder, action_decoder, caption_decoder):
        """
        Initialize unified model.

        Args:
            encoder: Shared ResNet101 encoder
            action_decoder: ActionDecoder for classification
            caption_decoder: DecoderWithAttention for caption generation
        """
        super(UnifiedActionCaptionModel, self).__init__()

        self.encoder = encoder
        self.action_decoder = action_decoder
        self.caption_decoder = caption_decoder

    def forward_action(self, images):
        """
        Forward pass for action recognition only.

        Args:
            images: Input images (batch_size, 3, 224, 224)

        Returns:
            logits: Classification logits (batch_size, 40)
            attention: Attention weights (batch_size, 49)
        """
        encoder_out = self.encoder(images)
        logits, attention = self.action_decoder(encoder_out)
        return logits, attention

    def forward_caption(self, images, captions, caption_lengths):
        """
        Forward pass for caption generation (training mode).

        Args:
            images: Input images (batch_size, 3, 224, 224)
            captions: Encoded captions (batch_size, max_len)
            caption_lengths: Caption lengths (batch_size,)

        Returns:
            predictions: Word predictions (batch_size, max_len, vocab_size)
            alphas: Attention weights
            encoded_captions: Sorted captions
            decode_lengths: Decode lengths
            sort_ind: Sort indices
        """
        encoder_out = self.encoder(images)
        predictions, alphas, encoded_captions, decode_lengths, sort_ind = \
            self.caption_decoder(encoder_out, captions, caption_lengths)
        return predictions, alphas, encoded_captions, decode_lengths, sort_ind

    def predict_combined(self, image, vocabulary, action_labels, device='cuda', top_k=5, max_caption_len=20):
        """
        Single inference: both action classification and caption generation.

        Args:
            image: Input image tensor (1, 3, 224, 224) or (3, 224, 224)
            vocabulary: Vocabulary object for caption decoding
            action_labels: ActionLabels object for action decoding
            device: Device
            top_k: Number of top action predictions
            max_caption_len: Maximum caption length

        Returns:
            Dictionary with combined results
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        image = image.to(device)

        self.eval()
        with torch.no_grad():
            # Encode image once (shared encoder)
            encoder_out = self.encoder(image)  # (1, 49, 512)

            # Action prediction
            action_logits, action_attention = self.action_decoder(encoder_out)
            action_probs = F.softmax(action_logits, dim=1)
            top_k_probs, top_k_indices = torch.topk(action_probs, min(top_k, action_probs.size(1)))

            # Caption generation (greedy decoding)
            caption = self._generate_caption_greedy(encoder_out, vocabulary, device, max_caption_len)

            # Format results
            result = {
                'predicted_class': top_k_indices[0, 0].item(),
                'action': action_labels.decode_action(top_k_indices[0, 0].item()),
                'confidence': top_k_probs[0, 0].item(),
                'caption': caption,
                'top_k': [
                    {
                        'action': action_labels.decode_action(idx.item()),
                        'confidence': prob.item()
                    }
                    for prob, idx in zip(top_k_probs[0], top_k_indices[0])
                ],
                'attention_map': action_attention[0].view(7, 7).cpu().numpy()  # Use action attention
            }

        return result

    def _generate_caption_greedy(self, encoder_out, vocabulary, device, max_length=20):
        """
        Generate caption using greedy decoding.

        Args:
            encoder_out: Encoder output (1, 49, 512)
            vocabulary: Vocabulary object
            device: Device
            max_length: Maximum caption length

        Returns:
            Generated caption string
        """
        try:
            # Initialize LSTM state
            h, c = self.caption_decoder.init_hidden_state(encoder_out)

            # Start with <START> token
            current_word = torch.LongTensor([vocabulary.word2idx['<START>']]).to(device)
            generated_words = []

            for _ in range(max_length):
                # Embed current word
                embeddings = self.caption_decoder.embedding(current_word).unsqueeze(1)
                embeddings = embeddings.squeeze(1)  # (1, embed_dim)

                # Attention
                context, alpha = self.caption_decoder.attention(encoder_out, h)

                # Gate context
                gate = torch.sigmoid(self.caption_decoder.f_beta(h))
                gated_context = gate * context

                # LSTM step
                h, c = self.caption_decoder.decode_step(
                    torch.cat([embeddings, gated_context], dim=1),
                    (h, c)
                )

                # Predict next word
                scores = self.caption_decoder.fc(h)
                predicted_idx = scores.argmax(dim=1).item()

                # Check for <END>
                if predicted_idx == vocabulary.word2idx['<END>']:
                    break

                # Add word (skip special tokens)
                word = vocabulary.idx2word.get(predicted_idx, '<UNK>')
                if word not in ['<PAD>', '<START>', '<END>', '<UNK>']:
                    generated_words.append(word)

                current_word = torch.LongTensor([predicted_idx]).to(device)

            return ' '.join(generated_words) if generated_words else "Unable to generate caption"

        except Exception as e:
            print(f"Caption generation error: {e}")
            return "Unable to generate caption"


def load_unified_model(action_checkpoint, caption_checkpoint, device='cuda'):
    """
    Load unified model from separate checkpoints.

    Args:
        action_checkpoint: Path to action model checkpoint
        caption_checkpoint: Path to caption model checkpoint
        device: Device to load on

    Returns:
        Unified model
    """
    from models.encoder import Encoder
    from models.action_decoder import ActionDecoder
    from models.caption_decoder import DecoderWithAttention

    print("Loading unified model...")

    # Load action checkpoint
    print(f"  Loading action model from {action_checkpoint}...")
    action_ckpt = torch.load(action_checkpoint, map_location=device)

    # Create encoder (shared)
    encoder = Encoder(encoded_size=512, fine_tune=False)

    # Create action decoder
    action_decoder = ActionDecoder(
        encoder_dim=512,
        decoder_dim=512,
        attention_dim=512,
        num_classes=40,
        lstm_steps=4,
        dropout=0.5
    )

    # Load full action model state dict
    model_state = action_ckpt['model_state_dict']

    # Extract encoder weights
    encoder_state = {
        k.replace('encoder.', ''): v
        for k, v in model_state.items()
        if k.startswith('encoder.')
    }
    encoder.load_state_dict(encoder_state)

    # Extract action decoder weights
    action_decoder_state = {
        k.replace('decoder.', ''): v
        for k, v in model_state.items()
        if k.startswith('decoder.')
    }
    action_decoder.load_state_dict(action_decoder_state)

    print(f"    ✓ Action model loaded")

    # Load caption checkpoint
    print(f"  Loading caption model from {caption_checkpoint}...")
    caption_ckpt = torch.load(caption_checkpoint, map_location=device)

    # Get vocab size from checkpoint
    vocab_size = caption_ckpt.get('vocab_size', 5004)

    # Create caption decoder
    caption_decoder = DecoderWithAttention(
        attention_dim=512,
        embed_dim=512,
        decoder_dim=512,
        vocab_size=vocab_size,
        encoder_dim=512,
        dropout=0.5
    )
    caption_decoder.load_state_dict(caption_ckpt['decoder_state_dict'])

    print(f"    ✓ Caption model loaded")

    # Create unified model
    model = UnifiedActionCaptionModel(encoder, action_decoder, caption_decoder)
    model.eval()
    model.to(device)

    print("✓ Unified model created and ready")

    return model


if __name__ == "__main__":
    """Test unified model."""
    print("=" * 80)
    print("Testing Unified Action-Caption Model")
    print("=" * 80)

    # Test model creation
    from models.encoder import Encoder
    from models.action_decoder import ActionDecoder
    from models.caption_decoder import DecoderWithAttention

    encoder = Encoder(encoded_size=512, fine_tune=False)
    action_decoder = ActionDecoder(num_classes=40)
    caption_decoder = DecoderWithAttention(vocab_size=5004)

    model = UnifiedActionCaptionModel(encoder, action_decoder, caption_decoder)

    print("\nModel components:")
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters()):,} params")
    print(f"  Action decoder: {sum(p.numel() for p in model.action_decoder.parameters()):,} params")
    print(f"  Caption decoder: {sum(p.numel() for p in model.caption_decoder.parameters()):,} params")
    print(f"  Total: {sum(p.numel() for p in model.parameters()):,} params")

    # Test forward passes
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)

    print("\nTesting action forward pass...")
    action_logits, action_attention = model.forward_action(images)
    print(f"  Action logits: {action_logits.shape}")
    print(f"  Action attention: {action_attention.shape}")

    print("\nTesting caption forward pass...")
    captions = torch.randint(0, 5004, (batch_size, 20))
    lengths = torch.randint(10, 20, (batch_size, 1))
    predictions, alphas, _, decode_lengths, _ = model.forward_caption(images, captions, lengths)
    print(f"  Predictions: {predictions.shape}")
    print(f"  Alphas: {alphas.shape}")

    print("\n" + "=" * 80)
    print("Unified model test completed successfully!")
    print("=" * 80)
