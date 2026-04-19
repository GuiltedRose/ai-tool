# pyscppy Integration Guide

## What You Need to Do

The only part of the code that's **incomplete** is the pyscppy data loading function. This is intentional because I don't have access to pyscppy's actual API documentation, but here's how to complete it.

## Current Placeholder Code

In `main.py`, around line 28, you'll see:

```python
def load_pyscppy_data():
    """Load EEG data from pyscppy."""
    if not HAS_PYSCPPY:
        raise ImportError("pyscppy is required. Install with: pip install pyscppy")
    
    # Load dataset from pyscppy
    # Adjust according to pyscppy's actual API
    data = pyscppy.load_dataset()  # or appropriate loading function
    x = data['x']  # EEG signals
    y = data['y']  # Labels
    
    return x, y
```

## What You Need to Find Out

After installing pyscppy (`pip install pyscppy`), follow these steps:

### Step 1: Explore pyscppy API

```python
import pyscppy
help(pyscppy)  # See available functions
dir(pyscppy)   # List all available functions/classes
```

Look for:
- Data loading functions: `load_data()`, `load_dataset()`, `get_dataset()`
- Classes: `Dataset`, `EEGDataset`, etc.
- Common return formats: dict, tuple, or class with `.data` and `.labels` attributes

### Step 2: Load Sample Data

Try different approaches:

```python
# Approach 1: Simple function
data = pyscppy.load_dataset()
print(data.keys())  # If it's a dict
print(type(data))

# Approach 2: Class-based
dataset = pyscppy.EEGDataset()
print(dir(dataset))

# Approach 3: Direct arrays
x, y = pyscppy.load_data()

# Approach 4: From file
data = pyscppy.load_from_file('path/to/data')
```

### Step 3: Check Data Shape

```python
import numpy as np

# After loading, verify format
print(f"X shape: {x.shape}")  # Should be (n_samples, ...) 
print(f"Y shape: {y.shape}")  # Should be (n_samples,)
print(f"X dtype: {x.dtype}")
print(f"Y unique: {np.unique(y)}")  # Class labels
```

### Step 4: Update the Function

Once you know the API, update `load_pyscppy_data()`:

**Example 1: If pyscppy has a load_dataset() function:**
```python
def load_pyscppy_data():
    """Load EEG data from pyscppy."""
    if not HAS_PYSCPPY:
        raise ImportError("pyscppy is required. Install with: pip install pyscppy")
    
    data = pyscppy.load_dataset()
    x = data['x']
    y = data['y']
    
    return x, y
```

**Example 2: If pyscppy returns a custom class:**
```python
def load_pyscppy_data():
    """Load EEG data from pyscppy."""
    if not HAS_PYSCPPY:
        raise ImportError("pyscppy is required. Install with: pip install pyscppy")
    
    dataset = pyscppy.EEGDataset()
    x = dataset.data
    y = dataset.labels
    
    return x, y
```

**Example 3: If pyscppy requires parameters:**
```python
def load_pyscppy_data():
    """Load EEG data from pyscppy."""
    if not HAS_PYSCPPY:
        raise ImportError("pyscppy is required. Install with: pip install pyscppy")
    
    # May need to specify subject, session, etc.
    x, y = pyscppy.load_data(subject='all', session='all')
    
    return x, y
```

## Testing Your Integration

After updating the function:

```python
# Test 1: Can you load data?
try:
    x, y = load_pyscppy_data()
    print("✓ Data loaded successfully")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Check data format
import numpy as np
print(f"X shape: {x.shape}")  # Should be (n_samples, ...)
print(f"Y shape: {y.shape}")  # Should be (n_samples,)
assert len(x) == len(y), "Mismatch between samples and labels"
print("✓ Data format correct")

# Test 3: Run full pipeline
python main.py
# Should start training and show metrics
```

## Common pyscppy Patterns

### If data is 1D time-series (n_samples, n_timepoints):
The network should adapt automatically, but verify reshaping:
```python
# In preprocessing, will become (n_samples, 1, 1, n_timepoints)
x.reshape(len(x), 1, 1, -1)
```

### If data is 2D (n_samples, n_channels, n_timepoints):
Update network input_shape:
```python
NETWORK_CONFIG['input_shape'] = (n_channels, n_channels, n_timepoints)
```

### If data includes metadata:
Extract just the signals and labels:
```python
x = data['signals']  # or data['eeg']
y = data['labels']   # or data['targets']
```

## Debug: Use data_explorer.py

Once you get data loading working, use:
```bash
python data_explorer.py
```

This will:
- Show data shape and statistics
- Visualize sample signals
- Check class distribution
- Test preprocessing steps

## Common Issues

**"pyscppy has no attribute 'load_dataset'"**
- Check available functions: `dir(pyscppy)`
- Look for: `load_data`, `load_eeg`, `get_dataset`, etc.

**"Shape mismatch in network"**
- Check actual shape: `print(x.shape, y.shape)`
- Update `NETWORK_CONFIG['input_shape']` accordingly

**"Data has NaN values"**
- Check: `np.isnan(x).sum()`
- May need cleaning in `preprocess_eeg_data()`

**"Too many classes"**
- List unique labels: `np.unique(y)`
- Network adjusts automatically via `to_categorical(y)`

## Resources

- pyscppy Documentation: (install and check with `help(pyscppy)`)
- scipy.signal: Used for convolution operations
- numpy: Array operations

## Once Working

1. ✅ pyscppy data loads correctly
2. ✅ Data shape verified
3. ✅ Preprocessing handles your data format
4. ✅ Network architecture configured for your dimensions
5. 🚀 Run `python main.py` to start training!

---

**Need help?** Check that pyscppy is installed: `pip list | grep pyscppy`
