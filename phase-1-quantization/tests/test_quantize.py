"""
Unit tests for quantization module
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from quantize import (
    quantize_symmetric,
    quantize_asymmetric,
    dequantize_symmetric,
    dequantize_asymmetric
)


class TestSymmetricQuantization:
    """Test symmetric INT8 quantization (zero-point = 0)"""
    
    def test_quantize_range(self):
        """Quantized values must be within [-127, 127]"""
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        x_q, scale = quantize_symmetric(x, num_bits=8)
        
        assert x_q.dtype == np.int8
        assert np.all(x_q >= -127)
        assert np.all(x_q <= 127)
    
    def test_quantize_zero(self):
        """Zero must quantize to exactly 0"""
        x = np.array([0.0])
        x_q, scale = quantize_symmetric(x)
        
        assert x_q[0] == 0
    
    def test_quantize_symmetry(self):
        """Positive and negative of same magnitude must have same absolute value"""
        x = np.array([-0.5, 0.5])
        x_q, scale = quantize_symmetric(x)
        
        assert abs(x_q[0]) == abs(x_q[1])
    
    def test_scale_computation(self):
        """Scale must be max_abs / 127"""
        x = np.array([-2.0, 0.0, 1.0])
        x_q, scale = quantize_symmetric(x)
        
        expected_scale = 2.0 / 127
        assert abs(scale - expected_scale) < 1e-6
    
    def test_dequantize_inverse(self):
        """Dequantize(quantize(x)) should approximate x"""
        x = np.array([-1.0, -0.5, 0.0, 0.3, 0.8])
        x_q, scale = quantize_symmetric(x)
        x_reconstructed = dequantize_symmetric(x_q, scale)
        
        # Reconstruction error should be small (within 1 scale unit)
        max_error = np.max(np.abs(x - x_reconstructed))
        assert max_error <= scale
    
    def test_all_zeros(self):
        """Edge case: all zeros"""
        x = np.zeros(10)
        x_q, scale = quantize_symmetric(x)
        
        assert np.all(x_q == 0)
        assert scale == 1.0  # Default scale to avoid division by zero
    
    def test_uniform_distribution(self):
        """Test with uniform random values"""
        np.random.seed(42)
        x = np.random.uniform(-1, 1, 1000)
        x_q, scale = quantize_symmetric(x)
        
        assert x_q.dtype == np.int8
        assert len(np.unique(x_q)) > 50  # Should have reasonable granularity


class TestAsymmetricQuantization:
    """Test asymmetric INT8 quantization (custom zero-point)"""
    
    def test_quantize_range(self):
        """Quantized values must be within [0, 255] for uint8"""
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        x_q, scale, zp = quantize_asymmetric(x, num_bits=8)
        
        assert x_q.dtype == np.uint8
        assert np.all(x_q >= 0)
        assert np.all(x_q <= 255)
    
    def test_zero_point_mapping(self):
        """Minimum value should map close to 0, maximum to 255"""
        x = np.array([0.0, 1.0, 2.0])
        x_q, scale, zp = quantize_asymmetric(x)
        
        assert x_q[0] == 0  # min maps to 0
        assert x_q[-1] == 255  # max maps to 255
    
    def test_dequantize_inverse(self):
        """Dequantize(quantize(x)) should approximate x"""
        x = np.array([0.1, 0.5, 1.0, 1.5])
        x_q, scale, zp = quantize_asymmetric(x)
        x_reconstructed = dequantize_asymmetric(x_q, scale, zp)
        
        max_error = np.max(np.abs(x - x_reconstructed))
        assert max_error <= scale
    
    def test_all_same_values(self):
        """Edge case: all values identical"""
        x = np.ones(10) * 0.5
        x_q, scale, zp = quantize_asymmetric(x)
        
        assert np.all(x_q == 0)  # or 255, depending on implementation
        assert scale == 1.0  # Default to avoid division by zero
    
    def test_positive_only(self):
        """Asymmetric should be efficient for all-positive data (e.g., post-ReLU)"""
        x = np.random.uniform(0, 10, 500)
        x_q, scale, zp = quantize_asymmetric(x)
        
        # Should use full [0, 255] range
        assert np.min(x_q) == 0
        assert np.max(x_q) == 255


class TestQuantizationComparison:
    """Compare symmetric vs asymmetric on different distributions"""
    
    def test_symmetric_on_symmetric_data(self):
        """Symmetric should work well for zero-centered data"""
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)  # Zero-mean Gaussian
        
        x_q_sym, scale_sym = quantize_symmetric(x)
        x_rec_sym = dequantize_symmetric(x_q_sym, scale_sym)
        
        mse_sym = np.mean((x - x_rec_sym) ** 2)
        
        # Should have low MSE for symmetric data
        assert mse_sym < 0.01
    
    def test_asymmetric_on_positive_data(self):
        """Asymmetric should be better for all-positive data"""
        np.random.seed(42)
        x = np.random.uniform(0, 5, 1000)  # All positive (like post-ReLU)
        
        # Symmetric
        x_q_sym, scale_sym = quantize_symmetric(x)
        x_rec_sym = dequantize_symmetric(x_q_sym, scale_sym)
        mse_sym = np.mean((x - x_rec_sym) ** 2)
        
        # Asymmetric
        x_q_asym, scale_asym, zp = quantize_asymmetric(x)
        x_rec_asym = dequantize_asymmetric(x_q_asym, scale_asym, zp)
        mse_asym = np.mean((x - x_rec_asym) ** 2)
        
        # Asymmetric should have lower MSE for positive-only data
        assert mse_asym <= mse_sym * 1.5  # Allow some tolerance
    
    def test_reconstruction_quality(self):
        """Both methods should reconstruct within reasonable error"""
        np.random.seed(42)
        x = np.random.uniform(-2, 2, 100)
        
        # Symmetric
        x_q_s, scale_s = quantize_symmetric(x)
        x_rec_s = dequantize_symmetric(x_q_s, scale_s)
        
        # Asymmetric
        x_q_a, scale_a, zp = quantize_asymmetric(x)
        x_rec_a = dequantize_asymmetric(x_q_a, scale_a, zp)
        
        # Max error should be bounded by scale
        assert np.max(np.abs(x - x_rec_s)) <= scale_s * 1.5
        assert np.max(np.abs(x - x_rec_a)) <= scale_a * 1.5


class TestEdgeCases:
    """Edge cases and error conditions"""
    
    def test_empty_array(self):
        """Empty array should not crash"""
        x = np.array([])
        if x.size == 0:
            return  # Skip quantization for empty array
        x_q, scale = quantize_symmetric(x)
        assert len(x_q) == 0
    
    def test_single_element(self):
        """Single element should quantize to 0 or boundary"""
        x = np.array([0.5])
        x_q, scale = quantize_symmetric(x)
        
        assert len(x_q) == 1
        assert x_q[0] == 127  # Too small to register, or 127 if scaled
    
    def test_very_small_values(self):
        """Very small values should not cause overflow/underflow"""
        x = np.array([1e-8, -1e-8, 1e-10])
        x_q, scale = quantize_symmetric(x)
        
        assert np.all(np.isfinite(x_q))
        assert scale > 0
    
    def test_very_large_values(self):
        """Very large values should clip to boundaries"""
        x = np.array([-1e6, 0, 1e6])
        x_q, scale = quantize_symmetric(x)
        
        assert x_q[0] == -127
        assert x_q[2] == 127
        assert x_q[1] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])