"""
Evaluation script: compare FP32 vs INT8 models
"""

import torch
import numpy as np
from torchvision import datasets, transforms
import json
import os


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    return obj


def evaluate_models():
    device = torch.device('cpu')
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    calib_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Import here to avoid circular imports
    from model import SimpleMLP
    from quantize import QuantizedMLP
    
    # FP32 model
    model_fp32 = SimpleMLP()
    model_fp32.load_state_dict(torch.load('results/float32/model_fp32.pth', map_location=device))
    model_fp32.eval()
    
    # Quantized model
    model_int8 = QuantizedMLP()
    model_int8.calibrate(calib_loader, num_samples=500)
    
    # Evaluate
    correct_fp32 = 0
    correct_int8 = 0
    total = 0
    errors = []
    
    for i, (data, target) in enumerate(test_loader):
        # FP32
        with torch.no_grad():
            output_fp32 = model_fp32(data)
            _, pred_fp32 = torch.max(output_fp32, 1)
            correct_fp32 += (pred_fp32 == target).item()
        
        # INT8
        output_int8 = model_int8.forward(data)
        pred_int8 = int(np.argmax(output_int8))
        correct_int8 += (pred_int8 == target.item())
        
        # Error analysis
        if pred_fp32.item() != pred_int8:
            errors.append({
                'index': int(i),
                'true': int(target.item()),
                'fp32_pred': int(pred_fp32.item()),
                'int8_pred': int(pred_int8),
                'fp32_conf': float(torch.softmax(output_fp32, dim=1).max().item()),
                'int8_conf': float(torch.softmax(torch.tensor(output_int8), dim=0).max().item())
            })
        
        total += 1
    
    accuracy_fp32 = 100.0 * correct_fp32 / total
    accuracy_int8 = 100.0 * correct_int8 / total
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"FP32 Accuracy: {accuracy_fp32:.2f}%")
    print(f"INT8 Symmetric Accuracy: {accuracy_int8:.2f}%")
    print(f"Degradation: {accuracy_fp32 - accuracy_int8:.2f}%")
    print(f"Disagreements: {len(errors)}/{total} ({100.0*len(errors)/total:.2f}%)")
    print(f"{'='*50}")
    
    # Save metrics
    os.makedirs('results/int8_symmetric', exist_ok=True)
    
    metrics = {
        'fp32_accuracy': accuracy_fp32,
        'int8_accuracy': accuracy_int8,
        'degradation': accuracy_fp32 - accuracy_int8,
        'disagreement_rate': 100.0 * len(errors) / total,
        'num_errors': len(errors),
        'error_examples': errors[:10]  # First 10 errors
    }
    
    # Convert to native Python types before JSON serialization
    metrics_native = convert_to_native(metrics)
    
    with open('results/int8_symmetric/metrics.json', 'w') as f:
        json.dump(metrics_native, f, indent=2)
    
    return metrics_native


if __name__ == '__main__':
    evaluate_models()