import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
    def __init__(self, kernel_size=5):
        super(BasicCNN, self).__init__()
        # First convolutional layer
        # Input: 1 x 28 x 28, Output: 16 x (28-kernel_size+1) x (28-kernel_size+1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=kernel_size)
        
        # Second convolutional layer
        # Input: 16 x ((28-kernel_size+1)/2) x ((28-kernel_size+1)/2) (after pooling)
        # Output: 32 x ((28-kernel_size+1)/2-kernel_size+1) x ((28-kernel_size+1)/2-kernel_size+1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size)
        
        # Calculate the size after convolutions and pooling
        conv1_out_size = 28 - kernel_size + 1
        pool1_out_size = conv1_out_size // 2
        conv2_out_size = pool1_out_size - kernel_size + 1
        pool2_out_size = conv2_out_size // 2
        
        # Fully connected layers
        # Dynamic calculation of features after convolutions and pooling
        self.fc1 = nn.Linear(32 * pool2_out_size * pool2_out_size, 128)
        self.fc2 = nn.Linear(128, 1)  # Binary classification
        
        # Store the final feature size for the forward pass
        self.final_features = 32 * pool2_out_size * pool2_out_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten - using the dynamically calculated size
        x = x.view(-1, self.final_features)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Sigmoid for binary classification
        return torch.sigmoid(x)
    
    def predict(self, x):
        """Return binary prediction based on threshold of 0.5"""
        with torch.no_grad():
            outputs = self(x)
            return (outputs >= 0.5).float()
