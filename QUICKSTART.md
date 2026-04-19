# Quick Start Guide - BCI Neural Network with biosppy

## Overview

Your neural network **automatically loads the BNCI Horizon 2020 EEG dataset** (motor imagery BCI data), similar to how MNIST is downloaded by Keras from a website. Zero setup required!

### How It Works

When you run `python main.py`:
1. ✅ Attempts to download BNCI Horizon 2020 dataset automatically
2. ✅ Falls back to local files (eeg_data.mat/npz) if download fails
3. ✅ Generates synthetic EEG data if local files not available
4. 🔢 Uses MNIST as final fallback for testing

## Quick Start

Just run:

```bash
pip install -r requirements.txt
python main.py
```

Expected output:
```
Attempting to load from BNCI Horizon 2020 dataset...
Downloading BNCI Horizon 2020 data...
Successfully loaded BNCI Horizon 2020 data
X shape: (288, 22, 1126), Y shape: (288,)
Applying biosppy preprocessing (fs=250Hz)...
Dataset split: Train=216, Val=43, Test=29
Starting training...
Epoch 10/500
  Train Loss: 0.234567 | Train Acc: 0.7500
  Val Loss: 0.245678 | Val Acc: 0.7200
```

**That's it!** Training starts automatically with real EEG data from BNCI Horizon 2020.

## Three Options

### 🚀 Option 1: Use BNCI Horizon 2020 (Recommended - Default)

No setup needed:
```bash
python main.py
```

Automatically downloads the BNCI Horizon 2020 motor imagery BCI dataset (288 samples, 22 channels, 1126 timepoints each).

### 📊 Option 2: Use Your Own Local EEG Data

If you have your own data:

1. Prepare as MATLAB (.mat) or NumPy (.npz) file with `x` and `y` arrays
2. Place in project root as `eeg_data.mat` or `eeg_data.npz`
3. Run:
   ```bash
   python main.py
   ```

The system automatically finds and loads your file instead of downloading BNCI.

### 🧪 Option 3: Use MNIST for Testing

If everything fails:
```bash
python main.py
# Automatically falls back to MNIST
```

## Data Formats (For Your Own Files)

### MATLAB Format (.mat)

```matlab
x = rand(100, 4, 250);  % 100 samples, 4 channels, 250 timepoints
y = randi([0, 3], 100, 1) - 1;  % Class labels (0, 1, 2, 3)
save('eeg_data.mat', 'x', 'y');
```

### NumPy Format (.npz)

```python
import numpy as np

x = np.random.rand(100, 4, 250)
y = np.random.randint(0, 4, 100)
np.savez('eeg_data.npz', x=x, y=y)
```

## Customization

Edit `config.py` to adjust:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `epochs` | 500 | Number of training iterations |
| `learning_rate` | 0.0001 | Weight update step size |
| `train_ratio` | 0.75 | Percentage for training |
| `val_ratio` | 0.15 | Percentage for validation |
| `test_ratio` | 0.10 | Percentage for testing |

Or modify directly in `main.py`:

```python
# Skip preprocessing (no bandpass filter)
(x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
    apply_preprocessing=False
)

# Custom data split (70/20/10)
(x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
    train_ratio=0.70,
    val_ratio=0.20,
    test_ratio=0.10
)

# Use specific local file
(x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
    data_file='path/to/my_data.mat'
)
```

## What Happens Automatically

1. **Data Loading**: Tries biosppy → local files → MNIST
2. **Preprocessing**: Bandpass filter (1-50 Hz) if enabled
3. **Normalization**: Z-score normalization
4. **Encoding**: One-hot encoding for labels
5. **Splitting**: 75% train / 15% val / 10% test
6. **Training**: Saves weights every epoch

## Troubleshooting

### "biosppy has no datasets"
- Normal! System falls back to local files or MNIST automatically

### "No data file found"
- Place `eeg_data.mat` or `eeg_data.npz` in project root, or
- Let it use MNIST fallback

### "Shape mismatch"
- Check your data: `print(x.shape, y.shape)`
- Update `config.py` `input_shape` if needed

### "Poor accuracy"
- Try different `learning_rate`: 0.001 or 0.00001
- Increase `epochs`: try 1000
- Check data is properly labeled

## File Reference

- **main.py** - Training script (run this!)
- **config.py** - Configuration parameters
- **data_explorer.py** - Analyze your data
- **BIOSPPY_GUIDE.md** - Detailed data format guide
- **layer.py, dense.py, convolution.py** - Neural network layers

## Next Steps

1. ✅ **Run immediately**: `python main.py`
2. 📊 **Explore data**: `python data_explorer.py`
3. ⚙️ **Customize**: Edit `config.py`
4. 🎓 **Provide own data**: Place file and rerun

---

**Ready? Just run:** `python main.py` 🚀
