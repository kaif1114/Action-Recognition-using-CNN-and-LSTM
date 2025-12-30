"""CNN Encoder for image feature extraction."""
import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """CNN Encoder using pre-trained ResNet101."""

    def __init__(self, encoded_size: int = 512, fine_tune: bool = True):
        """
        Initialize encoder.

        Args:
            encoded_size: Dimension of encoded image features
            fine_tune: If True, allow fine-tuning of CNN parameters
        """
        super(Encoder, self).__init__()

        self.encoded_size = encoded_size

        # Load pre-trained ResNet101
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        # Remove last two layers (avgpool and fc) to get spatial features
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pooling to get fixed size spatial features (7x7)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Linear layer to project features from 2048 to encoded_size
        self.fc = nn.Linear(2048, encoded_size)

        # Batch normalization (applied on feature dimension)
        self.bn = nn.BatchNorm1d(encoded_size, momentum=0.01)

        # ReLU activation
        self.relu = nn.ReLU(inplace=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Fine-tuning
        self.fine_tune(fine_tune)

    def forward(self, images):
        """
        Forward pass through encoder.

        Args:
            images: Input images (batch_size, 3, 224, 224)

        Returns:
            Encoded features (batch_size, num_pixels, encoded_size)
            where num_pixels = 49 (7x7 spatial grid)
        """
        # Extract features using ResNet
        features = self.resnet(images)  # (batch_size, 2048, H, W)

        # Adaptive pooling to get 7x7 spatial features
        features = self.adaptive_pool(features)  # (batch_size, 2048, 7, 7)

        # Get dimensions
        batch_size = features.size(0)

        # Reshape to (batch_size, 7, 7, 2048)
        features = features.permute(0, 2, 3, 1)

        # Reshape to (batch_size, 49, 2048)
        features = features.view(batch_size, -1, 2048)

        # Project to encoded_size
        features = self.fc(features)  # (batch_size, 49, encoded_size)

        # Apply batch normalization (on feature dimension)
        # BN expects (batch_size, num_features, sequence_length)
        features = features.permute(0, 2, 1)  # (batch_size, encoded_size, 49)
        features = self.bn(features)  # (batch_size, encoded_size, 49)
        features = features.permute(0, 2, 1)  # (batch_size, 49, encoded_size)

        # Apply ReLU and dropout
        features = self.relu(features)
        features = self.dropout(features)

        return features  # (batch_size, 49, encoded_size)

    def fine_tune(self, fine_tune: bool = True):
        """
        Enable or disable fine-tuning of CNN parameters.

        Args:
            fine_tune: If True, enable fine-tuning; otherwise freeze parameters
        """
        for param in self.resnet.parameters():
            param.requires_grad = fine_tune

        # Fine-tune only layers from layer2 onwards (keep earlier layers frozen)
        # This is a common practice to preserve low-level features
        if fine_tune:
            # Freeze first few layers
            for module in list(self.resnet.children())[:6]:
                for param in module.parameters():
                    param.requires_grad = False


class EncoderSimple(nn.Module):
    """
    Simplified encoder that outputs single feature vector per image.
    Useful for baseline model without attention.
    """

    def __init__(self, encoded_size: int = 512, fine_tune: bool = True):
        """
        Initialize simple encoder.

        Args:
            encoded_size: Dimension of encoded image features
            fine_tune: If True, allow fine-tuning of CNN parameters
        """
        super(EncoderSimple, self).__init__()

        self.encoded_size = encoded_size

        # Load pre-trained ResNet101
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        # Remove last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Linear layer to project from 2048 to encoded_size
        self.fc = nn.Linear(2048, encoded_size)

        # Batch normalization
        self.bn = nn.BatchNorm1d(encoded_size, momentum=0.01)

        # ReLU activation
        self.relu = nn.ReLU(inplace=False)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Fine-tuning
        self.fine_tune(fine_tune)

    def forward(self, images):
        """
        Forward pass through simple encoder.

        Args:
            images: Input images (batch_size, 3, 224, 224)

        Returns:
            Encoded features (batch_size, encoded_size)
        """
        # Extract features using ResNet
        features = self.resnet(images)  # (batch_size, 2048, 1, 1)

        # Flatten
        features = features.view(features.size(0), -1)  # (batch_size, 2048)

        # Project to encoded_size
        features = self.fc(features)  # (batch_size, encoded_size)

        # Apply batch normalization
        features = self.bn(features)

        # Apply ReLU and dropout
        features = self.relu(features)
        features = self.dropout(features)

        return features  # (batch_size, encoded_size)

    def fine_tune(self, fine_tune: bool = True):
        """Enable or disable fine-tuning of CNN parameters."""
        for param in self.resnet.parameters():
            param.requires_grad = fine_tune

        if fine_tune:
            # Freeze first few layers
            for module in list(self.resnet.children())[:6]:
                for param in module.parameters():
                    param.requires_grad = False


if __name__ == "__main__":
    # Test encoder
    encoder = Encoder(encoded_size=512, fine_tune=True)

    # Create dummy input
    dummy_images = torch.randn(4, 3, 224, 224)

    # Forward pass
    features = encoder(dummy_images)

    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: (4, 49, 512)")

    # Test simple encoder
    print("\nTesting simple encoder:")
    simple_encoder = EncoderSimple(encoded_size=512, fine_tune=True)
    simple_features = simple_encoder(dummy_images)
    print(f"Output shape: {simple_features.shape}")
    print(f"Expected: (4, 512)")
