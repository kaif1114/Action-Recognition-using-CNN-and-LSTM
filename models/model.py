"""Combined Encoder-Decoder model for action recognition."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.action_decoder import ActionDecoder


class EncoderDecoder(nn.Module):
    """Combined Encoder-Decoder model."""

    def __init__(self, encoder, decoder):
        """
        Initialize combined model.

        Args:
            encoder: Encoder instance
            decoder: Decoder instance
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, caption_lengths):
        """
        Forward pass during training.

        Args:
            images: Input images (batch_size, 3, 224, 224)
            captions: Encoded captions (batch_size, max_caption_length)
            caption_lengths: Caption lengths (batch_size,)

        Returns:
            predictions: Predicted scores (batch_size, max_caption_length, vocab_size)
            alphas: Attention weights (if decoder has attention)
            encoded_captions: Sorted captions
            decode_lengths: Decode lengths
            sort_ind: Sort indices
        """
        # Encode images
        encoder_out = self.encoder(images)

        # Decode captions
        predictions, alphas, encoded_captions, decode_lengths, sort_ind = self.decoder(
            encoder_out, captions, caption_lengths
        )

        return predictions, alphas, encoded_captions, decode_lengths, sort_ind

    def generate_caption(
        self,
        image,
        vocabulary,
        max_length=20,
        beam_size=1,
        device='cuda'
    ):
        """
        Generate caption for a single image.

        Args:
            image: Input image tensor (1, 3, 224, 224) or (3, 224, 224)
            vocabulary: Vocabulary object
            max_length: Maximum caption length
            beam_size: Beam size for beam search (1 for greedy)
            device: Device to run on

        Returns:
            caption: Generated caption string
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Move to device
        image = image.to(device)

        # Set model to eval mode
        self.eval()

        with torch.no_grad():
            if beam_size == 1:
                caption = self.greedy_decode(image, vocabulary, max_length, device)
            else:
                caption = self.beam_search(image, vocabulary, max_length, beam_size, device)

        return caption

    def greedy_decode(self, image, vocabulary, max_length, device):
        """
        Generate caption using greedy decoding.

        Args:
            image: Input image (1, 3, 224, 224)
            vocabulary: Vocabulary object
            max_length: Maximum caption length
            device: Device

        Returns:
            caption: Generated caption string
        """
        # Encode image
        encoder_out = self.encoder(image)  # (1, num_pixels, encoder_dim)

        # Initialize hidden state
        h, c = self.decoder.init_hidden_state(encoder_out)

        # Start token
        current_word = torch.tensor([vocabulary['<START>']]).to(device)

        # Generated caption
        generated_caption = []

        for _ in range(max_length):
            # Embed current word
            embeddings = self.decoder.embedding(current_word)  # (1, embed_dim)

            # Attention
            attention_weighted_encoding, alpha = self.decoder.attention(encoder_out, h)

            # Gate
            gate = torch.sigmoid(self.decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # LSTM step
            h, c = self.decoder.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c)
            )

            # Predict next word
            scores = self.decoder.fc(h)  # (1, vocab_size)
            predicted_word_idx = scores.argmax(dim=1).item()

            # Check for end token
            if predicted_word_idx == vocabulary['<END>']:
                break

            # Add to caption (skip unknown tokens in output)
            word = vocabulary.idx2word.get(predicted_word_idx, '<UNK>')
            if word not in ['<PAD>', '<START>', '<END>']:
                generated_caption.append(word)

            # Update current word
            current_word = torch.tensor([predicted_word_idx]).to(device)

        return ' '.join(generated_caption)

    def beam_search(self, image, vocabulary, max_length, beam_size, device):
        """
        Generate caption using beam search.

        Args:
            image: Input image (1, 3, 224, 224)
            vocabulary: Vocabulary object
            max_length: Maximum caption length
            beam_size: Beam size (number of hypotheses to maintain)
            device: Device

        Returns:
            caption: Generated caption string
        """
        # Encode image
        encoder_out = self.encoder(image)  # (1, num_pixels, encoder_dim)
        encoder_dim = encoder_out.size(-1)
        num_pixels = encoder_out.size(1)

        # Expand encoder output for beam search
        encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim)  # (beam_size, num_pixels, encoder_dim)

        # Initialize hidden state
        h, c = self.decoder.init_hidden_state(encoder_out)  # (beam_size, decoder_dim)

        # Start token for all beams
        k_prev_words = torch.LongTensor([[vocabulary['<START>']]] * beam_size).to(device)  # (beam_size, 1)

        # Scores for each sequence
        top_k_scores = torch.zeros(beam_size, 1).to(device)  # (beam_size, 1)

        # Lists to store completed sequences and their scores
        complete_seqs = []
        complete_seqs_scores = []

        # Start decoding
        step = 1

        while True:
            # Embed previous words
            embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (beam_size, embed_dim)

            # Attention
            attention_weighted_encoding, alpha = self.decoder.attention(encoder_out, h)

            # Gate
            gate = torch.sigmoid(self.decoder.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # LSTM step
            h, c = self.decoder.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c)
            )

            # Predict next word scores
            scores = self.decoder.fc(h)  # (beam_size, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add previous scores
            scores = top_k_scores.expand_as(scores) + scores  # (beam_size, vocab_size)

            # For first step, all k sequences will have the same score
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(beam_size, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)

            # Convert flattened indices to actual indices
            prev_word_inds = top_k_words // len(vocabulary)
            next_word_inds = top_k_words % len(vocabulary)

            # Build next sequences
            k_prev_words = torch.cat([k_prev_words[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            # Check for completed sequences
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != vocabulary['<END>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Add completed sequences
            if len(complete_inds) > 0:
                for ind in complete_inds:
                    complete_seqs.append(k_prev_words[ind].tolist())
                    complete_seqs_scores.append(top_k_scores[ind].item())

                beam_size -= len(complete_inds)

            # Stop if beam_size becomes 0 or max_length is reached
            if beam_size == 0 or step >= max_length:
                break

            # Update sequences for next iteration
            k_prev_words = k_prev_words[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

            step += 1

        # Select best sequence
        if len(complete_seqs_scores) > 0:
            best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
            best_seq = complete_seqs[best_idx]
        else:
            best_seq = k_prev_words[0].tolist()

        # Convert indices to words
        generated_caption = []
        for idx in best_seq:
            word = vocabulary.idx2word.get(idx, '<UNK>')
            if word not in ['<PAD>', '<START>', '<END>']:
                generated_caption.append(word)

        return ' '.join(generated_caption)


def create_model(vocab_size, device='cuda'):
    """
    Create and initialize encoder-decoder model.

    Args:
        vocab_size: Size of vocabulary
        device: Device to create model on

    Returns:
        model: EncoderDecoder model
    """
    # Model hyperparameters
    encoder_dim = 512
    attention_dim = 512
    embed_dim = 512
    decoder_dim = 512
    dropout = 0.5
    fine_tune = True

    # Create encoder
    encoder = Encoder(encoded_size=encoder_dim, fine_tune=fine_tune)

    # Create decoder
    decoder = DecoderWithAttention(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=encoder_dim,
        dropout=dropout
    )

    # Combine into single model
    model = EncoderDecoder(encoder, decoder)

    # Move to device
    model = model.to(device)

    return model


class ActionRecognitionModel(nn.Module):
    """
    Combined Encoder-Decoder model for action recognition.

    Unlike captioning, this model:
    - Takes only images as input (no captions)
    - Outputs classification logits (40 classes) instead of sequences
    - Returns attention weights for visualization
    """

    def __init__(self, encoder, decoder):
        """
        Initialize action recognition model.

        Args:
            encoder: Encoder instance (ResNet101)
            decoder: ActionDecoder instance
        """
        super(ActionRecognitionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images):
        """
        Forward pass for action recognition.

        Args:
            images: Input images (batch_size, 3, 224, 224)

        Returns:
            logits: Classification logits (batch_size, num_classes)
            attention_weights: Attention weights (batch_size, 49)
        """
        # Encode images to spatial features
        encoder_out = self.encoder(images)  # (batch_size, 49, 512)

        # Decode to action classification
        logits, attention_weights = self.decoder(encoder_out)
        # logits: (batch_size, 40)
        # attention_weights: (batch_size, 49)

        return logits, attention_weights

    def predict(self, images, action_labels, device='cuda', top_k=5):
        """
        Predict action with top-k results and attention visualization.

        Args:
            images: Input images (batch_size, 3, 224, 224) or (1, 3, 224, 224)
            action_labels: ActionLabels object for decoding
            device: Device for computation
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results for the first image in batch
        """
        # Ensure images have batch dimension
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        # Move to device
        images = images.to(device)

        # Set to eval mode
        self.eval()

        with torch.no_grad():
            logits, attention_weights = self.forward(images)
            probs = F.softmax(logits, dim=1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k)

        return {
            'predicted_class': top_k_indices[0, 0].item(),
            'action': action_labels.decode_action(top_k_indices[0, 0].item()),
            'confidence': top_k_probs[0, 0].item(),
            'top_k': [
                {
                    'action': action_labels.decode_action(idx.item()),
                    'confidence': prob.item()
                }
                for prob, idx in zip(top_k_probs[0], top_k_indices[0])
            ],
            'attention_map': attention_weights[0].view(7, 7).cpu().numpy()
        }


def create_action_model(num_classes=40, encoder_dim=512, decoder_dim=512,
                        lstm_steps=4, dropout=0.5, fine_tune=True, device='cuda'):
    """
    Create and initialize action recognition model.

    Args:
        num_classes: Number of action classes (40 for Stanford 40)
        encoder_dim: Encoder output dimension (512)
        decoder_dim: Decoder LSTM hidden dimension (512)
        lstm_steps: Number of LSTM steps (4)
        dropout: Dropout rate (0.5)
        fine_tune: Whether to fine-tune encoder (True)
        device: Device to create model on

    Returns:
        model: ActionRecognitionModel
    """
    # Create encoder (ResNet101)
    encoder = Encoder(encoded_size=encoder_dim, fine_tune=fine_tune)

    # Create action decoder
    decoder = ActionDecoder(
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        attention_dim=decoder_dim,
        num_classes=num_classes,
        lstm_steps=lstm_steps,
        dropout=dropout
    )

    # Combine into single model
    model = ActionRecognitionModel(encoder, decoder)

    # Move to device
    model = model.to(device)

    return model


if __name__ == "__main__":
    # Test model
    batch_size = 4
    vocab_size = 10000
    max_caption_length = 20

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(vocab_size, device=device)

    # Create dummy inputs
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    captions = torch.randint(0, vocab_size, (batch_size, max_caption_length)).to(device)
    caption_lengths = torch.randint(5, max_caption_length, (batch_size, 1)).to(device)

    # Forward pass
    print("Testing forward pass:")
    predictions, alphas, _, decode_lengths, _ = model(images, captions, caption_lengths)
    print(f"  Images shape: {images.shape}")
    print(f"  Captions shape: {captions.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Alphas shape: {alphas.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
