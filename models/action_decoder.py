"""Action Decoder for action recognition using LSTM with spatial attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Attention


class ActionDecoder(nn.Module):
    """
    LSTM decoder with spatial attention for action classification.

    Unlike caption decoders that generate variable-length sequences, this decoder:
    - Runs LSTM for fixed number of steps (no word embeddings as input)
    - Uses spatial attention to focus on relevant image regions
    - Outputs a single classification logit vector (40 classes)
    - Returns attention weights for visualization
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        decoder_dim: int = 512,
        attention_dim: int = 512,
        num_classes: int = 40,
        lstm_steps: int = 4,
        dropout: float = 0.5
    ):
        """
        Initialize action decoder.

        Args:
            encoder_dim: Dimension of encoder output features (512)
            decoder_dim: Dimension of LSTM hidden state (512)
            attention_dim: Dimension of attention layer (512)
            num_classes: Number of action classes (40 for Stanford 40)
            lstm_steps: Number of LSTM steps to run (4)
            dropout: Dropout rate (0.5)
        """
        super(ActionDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_classes = num_classes
        self.lstm_steps = lstm_steps

        # Attention mechanism
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # LSTM cell
        self.lstm = nn.LSTMCell(encoder_dim, decoder_dim)

        # Initialize LSTM state from encoder features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Attention gate (soft gating mechanism)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.fc = nn.Linear(decoder_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.init_h.weight)
        nn.init.xavier_uniform_(self.init_c.weight)
        nn.init.xavier_uniform_(self.f_beta.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        # Bias initialization
        nn.init.constant_(self.init_h.bias, 0)
        nn.init.constant_(self.init_c.bias, 0)
        nn.init.constant_(self.f_beta.bias, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, encoder_out):
        """
        Forward pass through action decoder.

        Args:
            encoder_out: Encoder output features (batch_size, num_pixels, encoder_dim)
                        For ResNet101: (batch_size, 49, 512)

        Returns:
            logits: Classification logits (batch_size, num_classes)
            attention_weights: Average attention weights (batch_size, num_pixels)
                             Can be reshaped to (batch_size, 7, 7) for heatmap
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        # Initialize LSTM state from mean-pooled encoder features
        # Mean pooling captures global image context
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch_size, encoder_dim)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)

        # Store attention weights for visualization
        attention_weights_all = []

        # Run LSTM for fixed number of steps with attention
        for step in range(self.lstm_steps):
            # Compute attention over encoder features
            context, alpha = self.attention(encoder_out, h)
            # context: (batch_size, encoder_dim)
            # alpha: (batch_size, num_pixels)

            # Store attention weights
            attention_weights_all.append(alpha)

            # Gate the context vector (soft attention gate)
            # This allows the model to control how much attention to use
            gate = torch.sigmoid(self.f_beta(h))  # (batch_size, encoder_dim)
            gated_context = gate * context  # (batch_size, encoder_dim)

            # LSTM step (avoid in-place operations for gradient flow)
            h_new, c_new = self.lstm(gated_context, (h, c))
            h = h_new
            c = c_new

        # Classification head
        # Use final hidden state for classification
        logits = self.fc(self.dropout(h))  # (batch_size, num_classes)

        # Average attention weights across all steps for visualization
        # This gives us a single attention map showing where the model looked
        attention_weights = torch.stack(attention_weights_all, dim=1).mean(dim=1)
        # attention_weights: (batch_size, num_pixels)

        return logits, attention_weights

    def predict(self, encoder_out, action_labels, device='cuda', top_k=5):
        """
        Predict action with top-k results and attention map.

        Args:
            encoder_out: Encoder output features (batch_size, 49, 512)
            action_labels: ActionLabels object for decoding
            device: Device for computation
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        self.eval()
        with torch.no_grad():
            logits, attention_weights = self.forward(encoder_out)
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


if __name__ == "__main__":
    # Test action decoder
    print("=" * 60)
    print("Testing Action Decoder")
    print("=" * 60)

    batch_size = 4
    num_pixels = 49  # 7x7 spatial grid from ResNet101
    encoder_dim = 512
    decoder_dim = 512
    num_classes = 40
    lstm_steps = 4

    # Create dummy encoder output
    encoder_out = torch.randn(batch_size, num_pixels, encoder_dim)
    print(f"\nInput encoder features: {encoder_out.shape}")

    # Create decoder
    decoder = ActionDecoder(
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        num_classes=num_classes,
        lstm_steps=lstm_steps
    )

    # Forward pass
    logits, attention_weights = decoder(encoder_out)

    print(f"\nOutput shapes:")
    print(f"  Logits: {logits.shape} (batch_size, num_classes)")
    print(f"  Attention weights: {attention_weights.shape} (batch_size, num_pixels)")
    print(f"\nLogits statistics:")
    print(f"  Min: {logits.min().item():.4f}")
    print(f"  Max: {logits.max().item():.4f}")
    print(f"  Mean: {logits.mean().item():.4f}")

    # Check attention weights sum to 1
    attention_sum = attention_weights.sum(dim=1)
    print(f"\nAttention weights sum (should be ~1.0):")
    print(f"  {attention_sum.tolist()}")

    # Test with softmax
    probs = F.softmax(logits, dim=1)
    print(f"\nProbabilities:")
    print(f"  Shape: {probs.shape}")
    print(f"  Sum per sample (should be 1.0): {probs.sum(dim=1).tolist()}")
    print(f"  Max probability: {probs.max(dim=1)[0].tolist()}")

    # Test attention heatmap reshape
    attention_heatmap = attention_weights[0].view(7, 7)
    print(f"\nAttention heatmap shape: {attention_heatmap.shape} (7x7 spatial grid)")

    print(f"\n{'=' * 60}")
    print("Action Decoder test completed successfully!")
    print("=" * 60)
