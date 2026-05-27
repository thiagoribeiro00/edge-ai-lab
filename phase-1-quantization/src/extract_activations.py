import torch
import numpy as np
from torchvision import datasets, transforms
from model import SimpleMLP

# Function to extract activations and weights from the trained model
def extract_activations(model_path='results/float32/model_fp32.pth', num_samples=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Hook to capture activations
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook
    
    # Register hooks for layers of interest
    model.fc1.register_forward_hook(hook_fn('fc1'))
    model.relu.register_forward_hook(hook_fn('relu'))
    model.fc2.register_forward_hook(hook_fn('fc2'))
    
    # Data preprocessing and loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load test dataset to extract activations from a subset of samples
    dataset = datasets.MNIST('./data', train=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialize dictionaries to store activations and weights
    all_activations = {'fc1': [], 'relu': [], 'fc2': []}
    all_weights = {
        'fc1_weight': model.fc1.weight.detach().cpu().numpy(),
        'fc1_bias': model.fc1.bias.detach().cpu().numpy(),
        'fc2_weight': model.fc2.weight.detach().cpu().numpy(),
        'fc2_bias': model.fc2.bias.detach().cpu().numpy()
    }
    
    # Extract activations for a subset of samples
    for i, (data, target) in enumerate(loader):
        if i >= num_samples:
            break
        data = data.to(device)
        with torch.no_grad():
            _ = model(data)
        
        # Append activations to the corresponding lists
        for key in all_activations:
            all_activations[key].append(activations[key].flatten())
    
    # Concatenate and save activations and weights to .npy files
    for key in all_activations:
        all_activations[key] = np.concatenate(all_activations[key])
    
    # Create results directory if it doesn't exist and save the data
    os.makedirs('results/float32', exist_ok=True)
    np.save('results/float32/activations.npy', all_activations)
    np.save('results/float32/weights.npy', all_weights)
    
    # Print statistics for weights and activations
    print("\n=== WEIGHTS STATISTICS ===")
    for name, w in all_weights.items():
        print(f"{name}: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}, std={w.std():.4f}")
    
    print("\n=== ACTIVATIONS STATISTICS ===")
    for name, a in all_activations.items():
        print(f"{name}: min={a.min():.4f}, max={a.max():.4f}, mean={a.mean():.4f}, std={a.std():.4f}")
    
    return all_weights, all_activations

if __name__ == '__main__':
    import os
    extract_activations()