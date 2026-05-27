# Phase 1: Manual INT8 Quantization

## Objective
Implement neural network quantization from scratch (no torch.quantization, no tensorflow.lite) to understand what happens under the hood of production quantization frameworks.

## What was done
- Trained simple MLP for MNIST (baseline FP32)
- Implemented symmetric INT8 quantization (zero-point = 0)
- Implemented asymmetric INT8 quantization (custom zero-point)
- Calibrated activation scales with representative validation data
- Simulated INT8 forward pass with dequantization-matmul-requantization pipeline

## Results

| Model | Accuracy | Degradation | Notes |
|-------|----------|-------------|-------|
| FP32 | 97.60% | — | Baseline |
| INT8 Symmetric | 97.65% | -0.05% | Negligible variance, quantization preserves accuracy |
| INT8 Asymmetric | TBD | TBD | Next step |

Key metrics:
- Disagreement rate: 0.13% (13/10,000 samples)
- Mean weight scale (FC1): ~0.0058 (derived from max/127)
- Activation range FC1: [-51.25, 36.65] — wide dynamic range, scale-sensitive

## What I learned
- Quantization is not just "multiply by scale" — the choice of symmetric vs asymmetric matters for activation distributions
- Post-ReLU activations are all-positive, making asymmetric quantization theoretically better (full 0-255 range used)
- Calibration data must be representative — using test data leaks information (fix: use train split)
- INT8 matmul in real hardware uses INT32 accumulation to prevent overflow — my simulation uses float intermediates for clarity

## How to run

```bash
# Train FP32 baseline
python src/train.py

# Extract weight and activation statistics
python src/extract_activations.py

# Evaluate FP32 vs INT8
python src/evaluate.py
```
## Project structure
```bash
phase-1-quantization/
├── src/
│   ├── model.py              # SimpleMLP definition
│   ├── train.py              # FP32 training script
│   ├── extract_activations.py # Stats extraction for calibration
│   ├── quantize.py           # INT8 quantization implementation
│   ├── evaluate.py           # Comparison and metrics
│   └── utils.py              # Helpers
├── results/
│   ├── float32/              # FP32 model, weights, metrics
│   └── int8_symmetric/       # INT8 metrics, error analysis
├── plots/                    # Generated visualizations
└── README.md
```

## Tests

Run unit tests for quantization functions:

```bash
pytest tests/test_quantize.py -v
```

### Next steps
[ ] Implement asymmetric quantization evaluation
[ ] Add per-channel quantization (per-output-channel for weights)
[ ] Quantization-aware training (QAT) — simulate quantization during training
[ ] Apply to vision model (CIFAR-10 with small CNN)
[ ] Measure inference speedup (simulated: count operations)