"""Attention mechanism for image captioning."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Bahdanau Attention mechanism.
    Computes attention weights over spatial features based on decoder hidden state.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        Initialize attention mechanism.

        Args:
            encoder_dim: Dimension of encoder output features
            decoder_dim: Dimension of decoder hidden state
            attention_dim: Dimension of attention layer
        """
        super(Attention, self).__init__()

        # Linear layer to transform encoder features
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)

        # Linear layer to transform decoder hidden state
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)

        # Linear layer to compute attention score
        self.full_att = nn.Linear(attention_dim, 1)

        # ReLU activation
        self.relu = nn.ReLU()

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass through attention mechanism.

        Args:
            encoder_out: Encoder output features (batch_size, num_pixels, encoder_dim)
            decoder_hidden: Decoder hidden state (batch_size, decoder_dim)

        Returns:
            context: Weighted sum of encoder features (batch_size, encoder_dim)
            alpha: Attention weights (batch_size, num_pixels)
        """
        # Transform encoder features
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)

        # Transform decoder hidden state
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)

        # Add broadcasted attention
        # att2.unsqueeze(1) -> (batch_size, 1, attention_dim)
        # Broadcasting makes it (batch_size, num_pixels, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)

        # Compute attention weights
        alpha = self.softmax(att)  # (batch_size, num_pixels)

        # Compute context vector as weighted sum
        # alpha.unsqueeze(2) -> (batch_size, num_pixels, 1)
        # encoder_out * alpha.unsqueeze(2) -> (batch_size, num_pixels, encoder_dim)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return context, alpha


class AdditiveAttention(nn.Module):
    """
    Alternative implementation of additive attention (same as Bahdanau).
    This is a more explicit implementation for educational purposes.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """Initialize additive attention."""
        super(AdditiveAttention, self).__init__()

        self.W_encoder = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass.

        Args:
            encoder_out: (batch_size, num_pixels, encoder_dim)
            decoder_hidden: (batch_size, decoder_dim)

        Returns:
            context: (batch_size, encoder_dim)
            alpha: (batch_size, num_pixels)
        """
        # Project encoder and decoder
        encoder_proj = self.W_encoder(encoder_out)  # (batch_size, num_pixels, attention_dim)
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)  # (batch_size, 1, attention_dim)

        # Compute attention scores
        scores = self.V(self.tanh(encoder_proj + decoder_proj)).squeeze(2)  # (batch_size, num_pixels)

        # Compute attention weights
        alpha = F.softmax(scores, dim=1)  # (batch_size, num_pixels)

        # Compute context
        context = torch.bmm(alpha.unsqueeze(1), encoder_out).squeeze(1)  # (batch_size, encoder_dim)

        return context, alpha


class DotProductAttention(nn.Module):
    """
    Dot-product attention (scaled).
    Faster but may be less expressive than additive attention.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int):
        """Initialize dot-product attention."""
        super(DotProductAttention, self).__init__()

        # Project decoder to encoder dimension if different
        self.decoder_proj = nn.Linear(decoder_dim, encoder_dim) if decoder_dim != encoder_dim else None

        # Scaling factor
        self.scale = encoder_dim ** 0.5

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass.

        Args:
            encoder_out: (batch_size, num_pixels, encoder_dim)
            decoder_hidden: (batch_size, decoder_dim)

        Returns:
            context: (batch_size, encoder_dim)
            alpha: (batch_size, num_pixels)
        """
        # Project decoder if needed
        if self.decoder_proj is not None:
            query = self.decoder_proj(decoder_hidden)  # (batch_size, encoder_dim)
        else:
            query = decoder_hidden

        # Compute dot product scores
        # query.unsqueeze(2) -> (batch_size, encoder_dim, 1)
        # encoder_out.transpose(1, 2) -> (batch_size, encoder_dim, num_pixels)
        scores = torch.bmm(encoder_out, query.unsqueeze(2)).squeeze(2)  # (batch_size, num_pixels)

        # Scale scores
        scores = scores / self.scale

        # Compute attention weights
        alpha = F.softmax(scores, dim=1)  # (batch_size, num_pixels)

        # Compute context
        context = torch.bmm(alpha.unsqueeze(1), encoder_out).squeeze(1)  # (batch_size, encoder_dim)

        return context, alpha


if __name__ == "__main__":
    # Test attention mechanism
    batch_size = 4
    num_pixels = 49  # 7x7 spatial grid
    encoder_dim = 512
    decoder_dim = 512
    attention_dim = 512

    # Create dummy inputs
    encoder_out = torch.randn(batch_size, num_pixels, encoder_dim)
    decoder_hidden = torch.randn(batch_size, decoder_dim)

    # Test Bahdanau attention
    print("Testing Bahdanau Attention:")
    attention = Attention(encoder_dim, decoder_dim, attention_dim)
    context, alpha = attention(encoder_out, decoder_hidden)
    print(f"  Encoder out shape: {encoder_out.shape}")
    print(f"  Decoder hidden shape: {decoder_hidden.shape}")
    print(f"  Context shape: {context.shape}")
    print(f"  Alpha shape: {alpha.shape}")
    print(f"  Alpha sum (should be 1): {alpha.sum(dim=1)}")

    # Test Additive attention
    print("\nTesting Additive Attention:")
    additive_att = AdditiveAttention(encoder_dim, decoder_dim, attention_dim)
    context2, alpha2 = additive_att(encoder_out, decoder_hidden)
    print(f"  Context shape: {context2.shape}")
    print(f"  Alpha shape: {alpha2.shape}")

    # Test Dot-product attention
    print("\nTesting Dot-Product Attention:")
    dot_att = DotProductAttention(encoder_dim, decoder_dim)
    context3, alpha3 = dot_att(encoder_out, decoder_hidden)
    print(f"  Context shape: {context3.shape}")
    print(f"  Alpha shape: {alpha3.shape}")
