"""Caption Decoder with Bahdanau Attention for image captioning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Attention


class DecoderWithAttention(nn.Module):
    """
    LSTM decoder with Bahdanau spatial attention for image caption generation.

    Unlike ActionDecoder which outputs classification logits, this decoder:
    - Takes word embeddings as input
    - Generates variable-length sequences
    - Outputs vocab_size logits for next word prediction
    - Uses teacher forcing during training
    """

    def __init__(
        self,
        attention_dim: int = 512,
        embed_dim: int = 512,
        decoder_dim: int = 512,
        vocab_size: int = 5004,
        encoder_dim: int = 512,
        dropout: float = 0.5
    ):
        """
        Initialize caption decoder.

        Args:
            attention_dim: Dimension of attention layer
            embed_dim: Dimension of word embeddings
            decoder_dim: Dimension of LSTM hidden state
            vocab_size: Size of vocabulary (5004 for COCO actions)
            encoder_dim: Dimension of encoder features (512 for ResNet101)
            dropout: Dropout rate
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout_rate = dropout

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Attention mechanism (reuse from models/attention.py)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # LSTM cell for decoding
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        # Initialize LSTM state from encoder features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Attention gate (soft gating mechanism)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection to vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Embedding initialization
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Linear layers - Xavier initialization
        nn.init.xavier_uniform_(self.init_h.weight)
        nn.init.xavier_uniform_(self.init_c.weight)
        nn.init.xavier_uniform_(self.f_beta.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        # Bias initialization
        nn.init.constant_(self.init_h.bias, 0)
        nn.init.constant_(self.init_c.bias, 0)
        nn.init.constant_(self.f_beta.bias, 0)
        nn.init.constant_(self.fc.bias, 0)

    def init_hidden_state(self, encoder_out):
        """
        Initialize LSTM hidden state and cell state from encoder features.

        Args:
            encoder_out: Encoder output features (batch_size, num_pixels, encoder_dim)

        Returns:
            h: Initial hidden state (batch_size, decoder_dim)
            c: Initial cell state (batch_size, decoder_dim)
        """
        # Mean pool encoder features to get global image representation
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch_size, encoder_dim)

        # Initialize h and c
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)

        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward pass during training (with teacher forcing).

        Args:
            encoder_out: Encoder features (batch_size, num_pixels, encoder_dim)
            encoded_captions: Encoded captions (batch_size, max_caption_length)
            caption_lengths: Caption lengths (batch_size,)

        Returns:
            predictions: Predicted word scores (batch_size, max_caption_length, vocab_size)
            alphas: Attention weights (batch_size, max_caption_length, num_pixels)
            encoded_captions: Sorted captions
            decode_lengths: Decode lengths (caption_lengths - 1)
            sort_ind: Sort indices
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        vocab_size = self.vocab_size

        # Sort input data by decreasing caption lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embed captions
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <END> position, since we've finished generating
        # So, decoding lengths = actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            # Determine batch size at this timestep (due to variable lengths)
            batch_size_t = sum([l > t for l in decode_lengths])

            # Compute attention over encoder output
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t]
            )

            # Gate the attention weighted encoding
            gate = torch.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # LSTM step
            h_new, c_new = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            # Update hidden states
            h = h_new
            c = c_new

            # Predict next word
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas, encoded_captions, decode_lengths, sort_ind


if __name__ == "__main__":
    """Test caption decoder."""
    print("=" * 80)
    print("Testing Caption Decoder with Attention")
    print("=" * 80)

    # Hyperparameters
    batch_size = 4
    num_pixels = 49  # 7x7 spatial grid from ResNet101
    encoder_dim = 512
    attention_dim = 512
    embed_dim = 512
    decoder_dim = 512
    vocab_size = 5004
    max_caption_len = 20

    # Create dummy inputs
    encoder_out = torch.randn(batch_size, num_pixels, encoder_dim)
    encoded_captions = torch.randint(0, vocab_size, (batch_size, max_caption_len))
    caption_lengths = torch.randint(10, max_caption_len, (batch_size, 1))

    print(f"\nInput shapes:")
    print(f"  Encoder out: {encoder_out.shape}")
    print(f"  Encoded captions: {encoded_captions.shape}")
    print(f"  Caption lengths: {caption_lengths.squeeze().tolist()}")

    # Create decoder
    decoder = DecoderWithAttention(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=encoder_dim,
        dropout=0.5
    )

    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Forward pass
    print(f"\nRunning forward pass...")
    decoder.train()
    predictions, alphas, sorted_captions, decode_lengths, sort_ind = decoder(
        encoder_out, encoded_captions, caption_lengths
    )

    print(f"\nOutput shapes:")
    print(f"  Predictions: {predictions.shape} (batch, max_decode_len, vocab_size)")
    print(f"  Alphas: {alphas.shape} (batch, max_decode_len, num_pixels)")
    print(f"  Sorted captions: {sorted_captions.shape}")
    print(f"  Decode lengths: {decode_lengths}")

    # Check predictions
    print(f"\nPrediction statistics:")
    print(f"  Min: {predictions.min().item():.4f}")
    print(f"  Max: {predictions.max().item():.4f}")
    print(f"  Mean: {predictions.mean().item():.4f}")

    # Check alphas sum to 1
    alpha_sums = alphas.sum(dim=2)  # Sum over num_pixels
    print(f"\nAttention weights sum (should be ~1.0):")
    print(f"  Example batch 0, timestep 0: {alpha_sums[0, 0].item():.4f}")
    print(f"  Example batch 0, timestep 5: {alpha_sums[0, 5].item():.4f}")

    # Test with softmax
    probs = F.softmax(predictions, dim=2)
    print(f"\nProbabilities:")
    print(f"  Shape: {probs.shape}")
    print(f"  Sum per word (should be 1.0): {probs[0, 0, :].sum().item():.4f}")
    print(f"  Max probability at t=0: {probs[0, 0, :].max().item():.4f}")

    # Test init_hidden_state
    print(f"\nTesting init_hidden_state:")
    h, c = decoder.init_hidden_state(encoder_out)
    print(f"  h shape: {h.shape} (batch_size, decoder_dim)")
    print(f"  c shape: {c.shape} (batch_size, decoder_dim)")

    print(f"\n{'=' * 80}")
    print("Caption Decoder test completed successfully!")
    print("=" * 80)
