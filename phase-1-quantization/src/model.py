import torch
import torch.nn as nn

# Define a simple MLP model for MNIST classification
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    # Define the forward pass
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten 28x28 -> 784
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x