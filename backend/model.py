"""
Hybrid CNN-RNN Model for Chickpea Disease Detection
Architecture based on research paper:
"Enhancing Chickpea Disease Detection Using Hybrid Deep Learning Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """
    CNN backbone for spatial feature extraction from leaf images.
    Extracts color, texture, and shape features of infected areas.
    """
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)

        # Block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.25)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        return x  # Shape: (batch, 128, H/8, W/8)


class HybridCNNRNN(nn.Module):
    """
    Hybrid CNN-RNN (LSTM) model as described in the research paper.

    Flow:
    1. CNN layers extract spatial features (color, texture, shape)
    2. Features are flattened into sequential vectors
    3. LSTM layers learn contextual/sequential dependencies
    4. Fully connected layer with Softmax for multi-class classification

    Disease Classes: Healthy, Blight, Rust, Wilt
    """

    def __init__(self, num_classes=4, image_size=128, hidden_size=256, num_lstm_layers=2):
        super(HybridCNNRNN, self).__init__()

        self.num_classes    = num_classes
        self.hidden_size    = hidden_size
        self.num_lstm_layers = num_lstm_layers

        # CNN Feature Extractor
        self.cnn = CNNFeatureExtractor()

        # Calculate CNN output size
        cnn_out_h = image_size // 8
        cnn_out_w = image_size // 8
        cnn_channels = 128

        # Sequence length = spatial positions, feature_size = channels
        self.seq_len      = cnn_out_h * cnn_out_w   # e.g. 16*16 = 256
        self.feature_size = cnn_channels              # 128

        # RNN (LSTM) for sequential/contextual learning
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Fully connected classification head
        # Bidirectional LSTM → hidden_size * 2
        self.fc1     = nn.Linear(hidden_size * 2, 256)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Step 1: CNN feature extraction → (batch, 128, H', W')
        cnn_features = self.cnn(x)

        # Step 2: Reshape for LSTM → (batch, seq_len, feature_size)
        # Flatten spatial dims: (batch, channels, H', W') → (batch, H'*W', channels)
        cnn_features = cnn_features.permute(0, 2, 3, 1)          # (B, H', W', C)
        cnn_features = cnn_features.reshape(batch_size, -1, self.feature_size)  # (B, seq, C)

        # Step 3: LSTM sequential learning
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)

        # Use last hidden state from both directions
        # h_n shape: (num_layers * 2, batch, hidden_size) for bidirectional
        h_forward  = h_n[-2]   # last forward layer
        h_backward = h_n[-1]   # last backward layer
        h_combined = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden*2)

        # Step 4: Classification
        out = F.relu(self.fc1(h_combined))
        out = self.drop_fc(out)
        out = self.fc2(out)   # logits

        return out            # (batch, num_classes)


# Disease class mapping
DISEASE_CLASSES = {
    0: "Healthy",
    1: "Blight",
    2: "Rust",
    3: "Wilt"
}

CLASS_DESCRIPTIONS = {
    "Healthy": "No disease detected. The plant appears healthy.",
    "Blight":  "Ascochyta Blight detected. Causes water-soaked lesions on leaves and stems.",
    "Rust":    "Chickpea Rust detected. Causes brown/orange pustules on leaf surfaces.",
    "Wilt":    "Fusarium Wilt detected. Causes yellowing and wilting of the entire plant."
}

CLASS_TREATMENTS = {
    "Healthy": [
        "Continue regular monitoring",
        "Maintain proper irrigation and fertilization",
        "Apply preventive fungicide spray every 2-3 weeks"
    ],
    "Blight": [
        "Remove and destroy infected plant parts immediately",
        "Apply Mancozeb or Chlorothalonil fungicide",
        "Avoid overhead irrigation to reduce leaf wetness",
        "Ensure proper crop spacing for air circulation"
    ],
    "Rust": [
        "Apply Propiconazole or Tebuconazole fungicide",
        "Remove heavily infected leaves and dispose off-field",
        "Avoid planting chickpea in the same field consecutively",
        "Use rust-resistant chickpea varieties in future planting"
    ],
    "Wilt": [
        "Remove and destroy wilted plants to prevent spread",
        "Apply Carbendazim soil drench around root zone",
        "Improve soil drainage to reduce moisture",
        "Practice crop rotation with non-legume crops for 2-3 years"
    ]
}


def get_model(num_classes=4, image_size=128):
    """Factory function to create the hybrid model."""
    return HybridCNNRNN(num_classes=num_classes, image_size=image_size)


if __name__ == "__main__":
    # Quick architecture test
    model = get_model()
    dummy  = torch.randn(2, 3, 128, 128)
    output = model(dummy)
    print(f"Model output shape: {output.shape}")
    print(f"Expected: (2, 4) — batch=2, classes=4")
    print("\nModel Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
