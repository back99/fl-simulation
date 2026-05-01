import torch.nn as nn

# Simple CNN model for MNIST digit classification (10 classes: 0-9)
# This model is shared between the server (global model) and each client (local training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # First convolutional layer: 1 input channel -> 32 feature maps
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce spatial size by half
            # Second convolutional layer: 32 -> 64 feature maps
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce spatial size by half again
            nn.Flatten(),     # Flatten to 1D vector for fully connected layers
            # Fully connected layers for final classification
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)  # Output: 10 class scores
        )

    def forward(self, x):
        return self.net(x)
