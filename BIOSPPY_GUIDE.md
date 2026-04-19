# biosppy Integration Guide

## Overview

Your neural network **automatically downloads the BNCI Horizon 2020 EEG dataset** (real motor imagery BCI data). Similar to how Keras downloads MNIST from a website, our system fetches authentic BCI data—no setup required!

The BNCI Horizon 2020 dataset includes:
- **288 samples** of motor imagery EEG data
- **22 EEG channels** 
- **1126 timepoints** per sample (4.5 seconds at 250 Hz)
- **4 classes** (motor imagery tasks)

### How It Works

When you run `python main.py`:

1. **First**, it attempts to download BNCI Horizon 2020 from the official server
2. **If download fails**, it looks for a local EEG data file (`eeg_data.mat` or `eeg_data.npz`)
3. **If no local file**, it generates synthetic EEG data automatically
4. **As a last resort**, it falls back to MNIST for testing

```
python main.py
    ↓
Try: Download BNCI Horizon 2020 (internet required)
    ↓ (if fails)
Try: Load local eeg_data.mat / eeg_data.npz
    ↓ (if not found)
Generate: Synthetic EEG data with biosppy preprocessing
    ↓ (if error)
Load MNIST (fallback - downloads from website)
```

## Quick Start

### Option 1: Use BNCI Horizon 2020 (Recommended)

```bash
pip install -r requirements.txt
python main.py
```

That's it! The system will automatically download the BNCI dataset and begin training.

**Expected output:**
```
Attempting to load from BNCI Horizon 2020 dataset...
Downloading BNCI Horizon 2020 data...
Successfully loaded BNCI Horizon 2020 data
X shape: (288, 22, 1126), Y shape: (288,)
Applying biosppy preprocessing (fs=250Hz)...
Dataset split: Train=216, Val=43, Test=29
Starting training...
```

### Option 2: Provide Your Own Local EEG Data

If you want to use custom EEG data instead of BNCI:

1. Prepare your EEG data file
2. Place it in project root as `eeg_data.mat` or `eeg_data.npz`
3. Run:
   ```bash
   python main.py
   ```

### Option 3: Use MNIST for Testing (Fallback)

If BNCI download fails and local files are unavailable:

```
Attempting to load from BNCI Horizon 2020 dataset... (fails - no internet)
No data file found...
Generating synthetic data... (optional)
Loading MNIST fallback data (downloaded from website)...
```

## About BNCI Horizon 2020

**BNCI Horizon 2020** is a standardized motor imagery BCI dataset from the Brain-Computer Interface Competition. It's widely used for BCI research and includes:

- **Real EEG data** from 9 subjects performing motor imagery
- **4 motor imagery tasks**: Left hand, Right hand, Both feet, Tongue
- **Recorded at 250 Hz** with 22 EEG channels
- **~1126 samples** per task (4.5 seconds each)
- **Bandpass filtered** 0.5-100 Hz with 50 Hz notch filter (in raw data)

This is the same dataset used in the BCI Competition IV.

## Synthetic Data Generation (Fallback)

If BNCI download fails, the system generates realistic synthetic EEG data with:

- **Alpha waves** (8-12 Hz) - varies by class
- **Beta waves** (12-30 Hz) - varies by channel  
- **Class modulation** - signal amplitude varies by class

This is generated using `scipy.signal` with biosppy's signal processing approach.

## Data Format (for local files)

If providing your own EEG data, use one of these formats:

### MATLAB Format (.mat)
- **X**: EEG signals with shape `(n_samples, n_channels, n_timepoints)` or `(n_samples, n_timepoints)`
- **Y**: Class labels with shape `(n_samples,)` containing integers 0, 1, 2, ...

Example MATLAB code to save your data:
```matlab
x = rand(100, 4, 250);  % 100 samples, 4 channels, 250 timepoints each
y = randi([0, 3], 100, 1) - 1;  % 4 classes (0, 1, 2, 3)
save('eeg_data.mat', 'x', 'y');
```

### NumPy Format (.npz)
- **x**: EEG signals as numpy array
- **y**: Class labels as numpy array

Example Python code to save your data:
```python
import numpy as np

x = np.random.rand(100, 4, 250)  % 100 samples, 4 channels, 250 timepoints
y = np.random.randint(0, 4, 100)  # 4 classes

np.savez('eeg_data.npz', x=x, y=y)
```

## What You Need to Do

### If Using Default BNCI Horizon 2020

**Nothing!** The system handles it automatically. Just run:
```bash
python main.py
```

The system will:
1. Automatically download BNCI Horizon 2020 (if internet available)
2. Fall back to local `eeg_data.mat` or `eeg_data.npz` if download fails
3. Generate synthetic data if local files not found
4. Use MNIST if all else fails

### If Providing Local Data

1. Prepare your EEG data in MATLAB or NumPy format (see above)
2. Save as `eeg_data.mat` or `eeg_data.npz` in project root:
   ```
   /Users/lunamoth/ai-tool/
   ├── main.py
   ├── eeg_data.mat    # <-- Your EEG data here
   └── ...
   ```
3. Run training:
   ```bash
   python main.py
   ```

## Using Your EEG Data

The system has four sources - it tries them in order:

### 1. Automatically Download from BNCI Horizon 2020 (Recommended)

```bash
python main.py
```

Downloads 288 samples of real motor imagery BCI data (22 channels, 1126 timepoints):
- Requires internet connection
- Automatic and transparent
- Real, validated BCI dataset
- ~30 MB download (happens once)

### 2. From Local File (If you have your own data)

Place your data file in project root and run:
```bash
# With MATLAB file
python main.py

# System will find and load 'eeg_data.mat' automatically
```

Or specify custom path in code:
```python
# In main.py, modify the load_and_split_data() call:
(x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
    data_file='path/to/your/data.mat',  # <-- Optional custom path
    apply_preprocessing=True
)
```

### 3. MNIST Fallback (For Testing)

If BNCI download fails and local files are unavailable:
```
Attempting to load from BNCI Horizon 2020 dataset... (fails - no internet)
No data file found...
Generating synthetic data... (error)
Loading MNIST fallback data (downloaded from website)
```

## Data Processing Pipeline

Your data (from any source) goes through:

```
Raw EEG Data (from biosppy, file, or MNIST)
       ↓
apply_biosppy_preprocessing()
  • Butterworth bandpass filter (1-50 Hz)
  • 4th order, zero-phase filtering
  • Applied per channel
       ↓
preprocess_eeg_data()
  • Z-score normalization
  • One-hot label encoding
  • Shape reformatting
       ↓
Train/Val/Test Split (75/15/10)
       ↓
Neural Network Training
```

## Monitoring Progress

Expected output:
```
```
Loaded EEG data from eeg_data.mat
X shape: (100, 4, 250), Y shape: (100,)
Applying biosppy preprocessing (fs=250Hz)...
Dataset split: Train=75, Val=15, Test=10
Starting training...
```

## Customizing Data Loading

### Automatic Loading (Default)

```python
# In main.py - this is what happens automatically:
(x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
    # Tries: BNCI download → local file → synthetic → MNIST (in that order)
    apply_preprocessing=True
)
```

### Load Specific Local File

```python
# Override to use a specific local file:
(x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
    data_file='path/to/your/eeg_data.mat',
    apply_preprocessing=True
)
```

### Skip Preprocessing

```python
# Load data without bandpass filtering:
(x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
    apply_preprocessing=False  # Skip the 1-50 Hz bandpass filter
)
```

### Custom Train/Val/Test Split

```python
# Change from default 75/15/10 to 70/15/15:
(x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    apply_preprocessing=True
)
```

## biosppy Preprocessing

The `apply_biosppy_preprocessing()` function applies to your EEG data:
- **Bandpass Filter**: 1-50 Hz (standard for EEG)
- **Filter Type**: Butterworth, 4th order
- **Phase**: Zero-phase (forward-backward filtering)
- **Per-Channel**: Each channel processed independently
- **Error Handling**: Gracefully skips problematic samples

## Preparing Your Own EEG Data (Fallback Option)

If you want to provide custom local data files instead of using biosppy:

### From EEGLAB (.set files)
```python
import scipy.io
import numpy as np

data = scipy.io.loadmat('your_eeglab_data.mat')
# Extract signals and labels
x = data['EEG']['data'][0, 0]  # Adjust based on your structure
y = data['labels'][0]
np.savez('eeg_data.npz', x=x, y=y)
```

### From CSV Files
```python
import numpy as np
import pandas as pd

# Assuming: columns are channels, rows are timepoints
data = pd.read_csv('eeg_data.csv')
x = data.values  # Shape: (n_timepoints, n_channels)
x = x.T  # Transpose: (n_channels, n_timepoints)
x = x[np.newaxis, :, :]  # Add sample dimension: (1, n_channels, n_timepoints)

y = np.array([...])  # Your class labels

np.savez('eeg_data.npz', x=x, y=y)
```

### From Raw Binary
```python
import numpy as np

# Load raw EEG binary
raw_data = np.fromfile('eeg_data.bin', dtype=np.float32)

# Reshape to (n_samples, n_channels, n_timepoints)
# Example: 1000 samples, 4 channels, 250 timepoints each
x = raw_data.reshape(1000, 4, 250)
y = np.array([...])  # Load your labels separately

np.savez('eeg_data.npz', x=x, y=y)
```

## Data Format Checklist

Before running training, verify your data:

- [ ] X shape is `(n_samples, n_channels, n_timepoints)` or `(n_samples, n_timepoints)`
- [ ] Y shape is `(n_samples,)` with integer class labels
- [ ] All samples have same number of channels and timepoints
- [ ] No NaN or Inf values: `np.any(np.isnan(x))` should be False
- [ ] Class labels are 0-indexed without gaps (e.g., 0, 1, 2, 3)
- [ ] File is in .mat or .npz format
- [ ] File is in the project root directory or provide full path

## Debugging Data Loading

### Check Data File
```python
import numpy as np
from scipy.io import loadmat

# For MATLAB files
data = loadmat('eeg_data.mat')
print("Keys in file:", data.keys())
print("X shape:", data['x'].shape)
print("Y shape:", data['y'].shape)

# For NPZ files
data = np.load('eeg_data.npz')
print("Keys in file:", data.files)
print("X shape:", data['x'].shape)
print("Y shape:", data['y'].shape)
```

### Test Preprocessing
```python
from main import load_biosppy_eeg_data, apply_biosppy_preprocessing

x, y = load_biosppy_eeg_data('eeg_data.mat')
print(f"Before preprocessing: {x.shape}, range [{x.min()}, {x.max()}]")

x_processed = apply_biosppy_preprocessing(x)
print(f"After preprocessing: {x_processed.shape}, range [{x_processed.min()}, {x_processed.max()}]")
```

## Quick Start

1. **Prepare your EEG data** in .mat or .npz format
2. **Place it in project directory** as `eeg_data.mat` or specify path
3. **Run training**:
   ```bash
   python main.py
   ```
4. **Monitor metrics**:
   - Train/Val/Test Loss
   - Train/Val/Test Accuracy
   - Training time and ETA

## Common Issues

### "Data file not found"
- Check file path is correct
- Verify file is in project root or provide absolute path
- Ensure filename matches: `eeg_data.mat` or update in code

### "Shape mismatch in network"
- Print actual data shape: `print(x.shape)`
- Update `NETWORK_CONFIG['input_shape']` in config.py
- Verify preprocessing doesn't change dimensions

### "Data has NaN values"
- Check source file: `np.any(np.isnan(x))`
- Try loading without preprocessing: `apply_preprocessing=False`
- Implement NaN handling in preprocessing

### "Poor accuracy on EEG data"
- Verify data is properly labeled
- Check for class imbalance: `np.bincount(y)`
- Try different network architectures
- Adjust learning rate and epochs

## Next Steps

1. **Prepare EEG data** file (see examples above)
2. **Verify data format** (shape, dtype, value ranges)
3. **Test loading** with debug code above
4. **Run training**: `python main.py`
5. **Tune hyperparameters** as needed

---

**Resources:**
- biosppy documentation: `pip show biosppy` or PyPI
- scipy.signal documentation: Signal processing functions
- Data format examples: See "Preparing Your Own EEG Data" section above
