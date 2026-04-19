# AI Neural Network Implementation for BCI Project

**👉 New to this project? Start with [QUICKSTART.md](QUICKSTART.md) for quick setup instructions!**

**📋 To use your EEG data? See [BIOSPPY_GUIDE.md](BIOSPPY_GUIDE.md) for detailed data preparation!**

## Current Architecture Overview

This is a custom neural network implementation built from scratch in Python (no PyTorch/TensorFlow for core layers) designed for EEG/BCI data classification. Supports both real EEG data via biosppy and MNIST (for testing).

### File Structure

- **main.py** - Main training loop; loads data (pyscppy or MNIST fallback), trains with 75/15/10 split
- **layer.py** - Base `Layer` class defining forward/backward interface
- **dense.py** - Fully connected (Dense) layer with weight updates
- **convolution.py** - 2D Convolutional layer using scipy's correlate2d/convolve2d
- **activation.py** - Activation functions: Sigmoid, Tanh, and Activate base class
- **losses.py** - Cross-entropy loss and its derivative
- **reshape.py** - Reshape layer for flattening data
- **test.py** - Testing utilities for signal processing
- **weights.npz** - Saved weights from training
- **requirements.txt** - Python dependencies including pyscppy

### Network Architecture (EEG-Adapted)

```
Input (1, 1, n_timepoints)
  ↓
Convolution (3x3 kernel, 5 filters)
  ↓
Sigmoid Activation
  ↓
Reshape to (flattened, 1)
  ↓
Dense (flattened → 100)
  ↓
Sigmoid Activation
  ↓
Dense (100 → n_classes)
  ↓
Softmax
  ↓
Output (n_classes)
```

### Training Details

- **Loss Function**: Cross-entropy
- **Epochs**: 500
- **Learning Rate**: 0.0001
- **Data Split**: 75% train / 15% validation / 10% test
- **Batch Size**: 1 (SGD - processes each sample individually)
- **Training Metrics**: Loss and accuracy tracked per epoch

## Using with pyscppy

### Installation

```bash
pip install -r requirements.txt
```

### Data Format Expected

The `load_pyscppy_data()` function expects pyscppy to provide:
- `x`: EEG signals of shape `(n_samples, n_features)` or `(n_samples, n_channels, n_timepoints)`
- `y`: Class labels of shape `(n_samples,)` with integer class indices

### Preprocessing

- **Normalization**: Z-score normalization (zero mean, unit variance)
- **Encoding**: One-hot encoding of class labels
- **Reshaping**: Adapted for network input compatibility

### Adjusting Network for Your Data

Modify the network architecture in `main.py` based on your EEG data shape:

```python
# Example: if EEG data is (n_samples, 128) - 128 channels/features
# Input shape after preprocessing: (1, 1, 128)
network = [
    Convolution((1, 1, 128), 3, 5),     # Input shape, kernel size, filters
    Sigmoid(),
    Reshape((5, 1, 126), (5 * 126, 1)), # Output shape of conv, new shape
    Dense(5 * 126, 100),
    Sigmoid(),
    Dense(100, n_classes),
    Softmax()
]
```

### Running Training

```bash
python main.py
```

Expected output:
```
Starting training...
Train samples: 750, Val samples: 150, Test samples: 100
Epoch 10/500
  Train Loss: 0.123456 | Train Acc: 0.8500
  Val Loss: 0.145678 | Val Acc: 0.8200
  ...
```

## Customization Guide

### Changing the Data Split

Edit `load_and_split_data()` parameters:
```python
(x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
    train_ratio=0.75, 
    val_ratio=0.15, 
    test_ratio=0.10
)
```

### Adjusting Learning Parameters

```python
epochs = 500           # Number of training iterations
learning_rate = 0.0001 # Step size for weight updates
```

### Changing Network Depth

Add more Dense layers for a deeper network:
```python
network = [
    Convolution((1, 1, 128), 3, 5),
    Sigmoid(),
    Reshape((5, 1, 126), (630, 1)),
    Dense(630, 256),      # Hidden layer 1
    Sigmoid(),
    Dense(256, 128),      # Hidden layer 2
    Sigmoid(),
    Dense(128, n_classes),
    Softmax()
]
```

## Key Dependencies

- **numpy**: Array operations
- **scipy**: Convolution operations
- **tensorflow**: Data utilities (to_categorical)
- **pyscppy**: EEG dataset loading
- **biosppy**: Signal processing (optional, for filtering)

## Fallback Behavior

If pyscppy is not installed or dataset fails to load, the system automatically falls back to MNIST for testing purposes. This allows you to verify the network architecture works before integrating real EEG data.

## Next Steps for BCI Integration

1. **Verify pyscppy Data Format**: Test `load_pyscppy_data()` output shape
2. **Optimize Network Architecture**: Adjust conv layers, filters, and dense layers for best performance
3. **Add Signal Preprocessing**: Use biosppy for filtering (bandpass, notch filters)
4. **Hyperparameter Tuning**: Experiment with learning rate, regularization
5. **Cross-validation**: Consider k-fold cross-validation for better validation metrics

