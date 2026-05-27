import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleMLP
import json
import os

# Training function for the SimpleMLP model on the MNIST dataset using FP32 precision
def train_model(epochs=10, batch_size=64, lr=0.001):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data preprocessing and loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset for training and testing
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    
    # Load test dataset for evaluation after training
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=batch_size, shuffle=False
    )
    
    # Model and training setup
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with FP32 precision
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    # Use torch.no_grad() to disable gradient calculations during evaluation for efficiency
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # Calculate and print the test accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy (FP32): {accuracy:.2f}%')
    
    # Save model and metrics to 'results/float32' directory
    os.makedirs('results/float32', exist_ok=True)
    torch.save(model.state_dict(), 'results/float32/model_fp32.pth')
    
    # Save metrics to a JSON file for later analysis
    metrics = {
        'accuracy': accuracy,
        'train_losses': train_losses,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }
    
    # Write the metrics to a JSON file with indentation for readability
    with open('results/float32/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return model, metrics

print("Training function defined successfully.")

# Run the training function when the script is executed directly
if __name__ == '__main__':
    model, metrics = train_model()