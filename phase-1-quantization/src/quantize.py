"""
Manual INT8 Quantization Implementation
No torch.quantization, no tensorflow.lite - just numpy
"""

import numpy as np
import torch


def quantize_symmetric(x, num_bits=8):
    """
    Symmetric quantization: zero-point = 0
    Range: [-127, 127] for 8 bits (reserve 1 bit for sign)
    """
    qmin = -(2 ** (num_bits - 1) - 1)  # -127
    qmax = 2 ** (num_bits - 1) - 1     # 127
    
    abs_max = np.max(np.abs(x)) if x.size > 0 else 0.0
    if abs_max == 0:
        scale = 1.0
    else:
        scale = abs_max / qmax
    
    x_quant = np.clip(np.round(x / scale), qmin, qmax).astype(np.int8)
    return x_quant, scale


def dequantize_symmetric(x_quant, scale):
    """Dequantize back to float"""
    return x_quant.astype(np.float32) * scale


def quantize_asymmetric(x, num_bits=8):
    """
    Asymmetric quantization: custom zero-point
    Range: [0, 255] mapped to [min, max]
    """
    qmin = 0
    qmax = 2 ** num_bits - 1  # 255
    
    x_min = np.min(x)
    x_max = np.max(x)
    
    if x_max == x_min:
        scale = 1.0
        zero_point = 0
    else:
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = np.round(qmin - x_min / scale).astype(np.int32)
    
    x_quant = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.uint8)
    return x_quant, scale, zero_point


def dequantize_asymmetric(x_quant, scale, zero_point):
    """Dequantize back to float"""
    return (x_quant.astype(np.float32) - zero_point) * scale


class QuantizedMLP:
    """
    MLP running with quantized weights and activations
    Simulates what happens in INT8 hardware
    """
    
    def __init__(self, model_path='results/float32/model_fp32.pth'):
        # Load FP32 model
        device = torch.device('cpu')
        from model import SimpleMLP
        model = SimpleMLP()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Extract weights
        self.fc1_weight = model.fc1.weight.detach().cpu().numpy()
        self.fc1_bias = model.fc1.bias.detach().cpu().numpy()
        self.fc2_weight = model.fc2.weight.detach().cpu().numpy()
        self.fc2_bias = model.fc2.bias.detach().cpu().numpy()
        
        # Quantize weights (offline calibration)
        self.fc1_weight_q, self.fc1_weight_scale = quantize_symmetric(self.fc1_weight)
        self.fc2_weight_q, self.fc2_weight_scale = quantize_symmetric(self.fc2_weight)
        
        # Bias stays in float for simplicity, or quantize to INT32
        # For accurate simulation, we keep bias in float and add after dequantization
        
        # Activation scales (calibrated with validation data)
        self.fc1_act_scale = None
        self.relu_act_scale = None
        self.fc2_act_scale = None
    
    def calibrate(self, dataloader, num_samples=500):
        """Calibrate activation scales with representative data"""
        from model import SimpleMLP
        import torch
        
        model = SimpleMLP()
        model.load_state_dict(torch.load('results/float32/model_fp32.pth'))
        model.eval()
        
        fc1_acts = []
        relu_acts = []
        fc2_acts = []
        
        for i, (data, _) in enumerate(dataloader):
            if i >= num_samples:
                break
            with torch.no_grad():
                x = data.view(data.size(0), -1)
                a1 = model.fc1(x).numpy().flatten()
                r1 = model.relu(model.fc1(x)).numpy().flatten()
                a2 = model.fc2(model.relu(model.fc1(x))).numpy().flatten()
                
                fc1_acts.extend(a1)
                relu_acts.extend(r1)
                fc2_acts.extend(a2)
        
        self.fc1_act_scale = quantize_symmetric(np.array(fc1_acts))[1]
        self.relu_act_scale = quantize_symmetric(np.array(relu_acts))[1]
        self.fc2_act_scale = quantize_symmetric(np.array(fc2_acts))[1]
    
    def forward(self, x):
        """
        Forward pass simulating INT8:
        1. Quantize input
        2. Simulated INT8 matmul (simplified: dequantize to float, matmul, requantize)
        3. ReLU (in float, then requantize)
        4. Final INT8 simulated matmul
        """
        # Convert torch tensor to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        
        # Flatten input
        x_flat = x.flatten()
        
        # Quantize input
        x_q, x_scale = quantize_symmetric(x_flat)
        
        # FC1: simulated INT8 matmul
        # In real hardware: INT32 accumulation with rescale
        # Here: dequantize to float, matmul, add bias
        fc1_out = np.dot(x_q.astype(np.float32), self.fc1_weight_q.T.astype(np.float32))
        fc1_out = fc1_out * x_scale * self.fc1_weight_scale
        fc1_out += self.fc1_bias
        
        # ReLU
        fc1_out = np.maximum(0, fc1_out)
        
        # Requantize to INT8 after ReLU
        fc1_out_q, fc1_out_scale = quantize_symmetric(fc1_out)
        
        # FC2: simulated INT8 matmul
        fc2_out = np.dot(fc1_out_q.astype(np.float32), self.fc2_weight_q.T.astype(np.float32))
        fc2_out = fc2_out * fc1_out_scale * self.fc2_weight_scale
        fc2_out += self.fc2_bias
        
        return fc2_out


class QuantizedMLP_Asymmetric:
    """
    MLP with asymmetric quantization
    Better for activations with non-symmetric distributions (like after ReLU)
    """
    
    def __init__(self, model_path='results/float32/model_fp32.pth'):
        # Load FP32 model
        device = torch.device('cpu')
        from model import SimpleMLP
        model = SimpleMLP()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Extract weights
        self.fc1_weight = model.fc1.weight.detach().cpu().numpy()
        self.fc1_bias = model.fc1.bias.detach().cpu().numpy()
        self.fc2_weight = model.fc2.weight.detach().cpu().numpy()
        self.fc2_bias = model.fc2.bias.detach().cpu().numpy()
        
        # Quantize weights with symmetric (weights are usually symmetric around 0)
        self.fc1_weight_q, self.fc1_weight_scale = quantize_symmetric(self.fc1_weight)
        self.fc2_weight_q, self.fc2_weight_scale = quantize_symmetric(self.fc2_weight)
        
        # Activation scales (asymmetric for post-ReLU)
        self.fc1_act_scale = None
        self.fc1_act_zp = None
        self.relu_act_scale = None
        self.relu_act_zp = None
        self.fc2_act_scale = None
        self.fc2_act_zp = None
    
    def calibrate(self, dataloader, num_samples=500):
        """Calibrate activation scales with representative data"""
        from model import SimpleMLP
        import torch
        
        model = SimpleMLP()
        model.load_state_dict(torch.load('results/float32/model_fp32.pth'))
        model.eval()
        
        fc1_acts = []
        relu_acts = []
        fc2_acts = []
        
        for i, (data, _) in enumerate(dataloader):
            if i >= num_samples:
                break
            with torch.no_grad():
                x = data.view(data.size(0), -1)
                a1 = model.fc1(x).numpy().flatten()
                r1 = model.relu(model.fc1(x)).numpy().flatten()
                a2 = model.fc2(model.relu(model.fc1(x))).numpy().flatten()
                
                fc1_acts.extend(a1)
                relu_acts.extend(r1)
                fc2_acts.extend(a2)
        
        # Asymmetric for activations (especially post-ReLU which is all positive)
        self.fc1_act_scale, self.fc1_act_zp = quantize_asymmetric(np.array(fc1_acts))[1:]
        self.relu_act_scale, self.relu_act_zp = quantize_asymmetric(np.array(relu_acts))[1:]
        self.fc2_act_scale, self.fc2_act_zp = quantize_asymmetric(np.array(fc2_acts))[1:]
    
    def forward(self, x):
        """Forward pass with asymmetric quantization for activations"""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        
        x_flat = x.flatten()
        x_q, x_scale = quantize_symmetric(x_flat)  # Input still symmetric (normalized around 0)
        
        # FC1
        fc1_out = np.dot(x_q.astype(np.float32), self.fc1_weight_q.T.astype(np.float32))
        fc1_out = fc1_out * x_scale * self.fc1_weight_scale
        fc1_out += self.fc1_bias
        
        # ReLU
        fc1_out = np.maximum(0, fc1_out)
        
        # Requantize asymmetric after ReLU
        fc1_out_q, fc1_out_scale, fc1_out_zp = quantize_asymmetric(fc1_out)
        
        # FC2
        fc2_out = np.dot(fc1_out_q.astype(np.float32), self.fc2_weight_q.T.astype(np.float32))
        fc2_out = fc2_out * fc1_out_scale * self.fc2_weight_scale
        fc2_out += self.fc2_bias
        
        return fc2_out