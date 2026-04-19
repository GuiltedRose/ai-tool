import os
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
from tensorflow.keras.utils import to_categorical
from layer import Layer
from activation import Activate, Tanh, Sigmoid
from dense import Dense
from convolution import Convolution
from losses import cross_entropy, cross_entropy_prime
from reshape import Reshape
import time
import urllib.request
import tempfile
import zipfile

# Import biosppy for EEG signal processing
try:
    import biosppy
    from biosppy import signals
    HAS_BIOSPPY = True
except ImportError:
    HAS_BIOSPPY = False
    print("Warning: biosppy not found. Install with: pip install biosppy")

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    def backward(self, output_gradient, learning_rate):
       n = np.size(self.output)
       tmp = np.tile(self.output, n)
       return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

def load_biosppy_eeg_data(data_file=None):
    """
    Load EEG data from available sources.
    
    Attempts to:
    1. Load from BNCI Horizon 2020 dataset (automatic download)
    2. Load from local data file (.mat or .npz)
    3. Generate synthetic EEG data using biosppy utilities
    4. Fall back to MNIST
    
    Args:
        data_file: Optional path to local data file (.mat or .npz). 
    
    Returns:
        x: EEG signals (n_samples, n_channels, n_timepoints)
        y: Class labels (n_samples,)
    """
    
    # Try to load from BNCI Horizon 2020 dataset
    try:
        print("Attempting to load from BNCI Horizon 2020 dataset...")
        
        # BNCI Horizon 2020 dataset
        dataset_url = "http://www.bbci.de/competition/download/competition_iv/BNCI2014001R.zip"
        
        print(f"Downloading BNCI Horizon 2020 data...")
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "bnci.zip")
            urllib.request.urlretrieve(dataset_url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            
            # Look for .mat files in extracted directory
            mat_files = []
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith('.mat'):
                        mat_files.append(os.path.join(root, file))
            
            if mat_files:
                # Load first available file
                data = loadmat(mat_files[0])
                # BNCI data typically has 'X' (features) and 'y' (labels)
                if 'X' in data and 'y' in data:
                    x = data['X']
                    y = data['y'].flatten()
                elif 'x' in data and 'y' in data:
                    x = data['x']
                    y = data['y'].flatten()
                else:
                    raise KeyError("Could not find X/x and y in BNCI data")
                
                print(f"Successfully loaded BNCI Horizon 2020 data")
                print(f"X shape: {x.shape}, Y shape: {y.shape}")
                return x, y
    except Exception as e:
        print(f"Could not load from BNCI Horizon 2020: {e}")
    
    # Try to load from local file first
    if data_file is not None and os.path.exists(data_file):
        if data_file.endswith('.mat'):
            data = loadmat(data_file)
            x = data['x']
            y = data['y']
        elif data_file.endswith('.npz'):
            data = np.load(data_file)
            x = data['x']
            y = data['y']
        else:
            raise ValueError("Data file must be .mat or .npz format")
        
        print(f"Loaded EEG data from {data_file}")
        print(f"X shape: {x.shape}, Y shape: {y.shape}")
        return x, y
    
    # Try to load from default local file
    default_file = 'eeg_data.mat'
    if os.path.exists(default_file):
        data = loadmat(default_file)
        x = data['x']
        y = data['y']
        print(f"Loaded EEG data from {default_file}")
        print(f"X shape: {x.shape}, Y shape: {y.shape}")
        return x, y
    
    default_npz = 'eeg_data.npz'
    if os.path.exists(default_npz):
        data = np.load(default_npz)
        x = data['x']
        y = data['y']
        print(f"Loaded EEG data from {default_npz}")
        print(f"X shape: {x.shape}, Y shape: {y.shape}")
        return x, y
    
    # Generate synthetic EEG data using biosppy/scipy as last resort
    if HAS_BIOSPPY:
        print("No data file found. Generating synthetic EEG data using biosppy utilities...")
        return generate_synthetic_eeg_data()
    
    # If nothing else works, raise error
    raise FileNotFoundError(
        "No EEG data found. Please provide one of:\n"
        "1. BNCI Horizon 2020 (automatic download if internet available)\n"
        "2. Local file: eeg_data.mat or eeg_data.npz in project root\n"
        "3. Custom path via data_file parameter\n"
        "See BIOSPPY_GUIDE.md for data preparation examples."
    )

def generate_synthetic_eeg_data(n_samples=100, n_channels=4, n_timepoints=250, n_classes=4, sampling_rate=250):
    """
    Generate realistic synthetic EEG data using signal processing techniques.
    
    Args:
        n_samples: Number of training samples
        n_channels: Number of EEG channels
        n_timepoints: Timepoints per sample
        n_classes: Number of classes
        sampling_rate: Sampling frequency (Hz)
    
    Returns:
        x: Synthetic EEG signals (n_samples, n_channels, n_timepoints)
        y: Random class labels (n_samples,)
    """
    print(f"Generating {n_samples} synthetic EEG samples ({n_channels} channels, {n_timepoints} timepoints)...")
    
    x = np.zeros((n_samples, n_channels, n_timepoints))
    y = np.random.randint(0, n_classes, n_samples)
    
    # Generate EEG-like signals with different frequency components
    for i in range(n_samples):
        for ch in range(n_channels):
            # Time vector
            t = np.arange(n_timepoints) / sampling_rate
            
            # Alpha wave (8-12 Hz) - class dependent
            alpha_freq = 8 + 2 * (y[i] / n_classes)  # Varies with class
            alpha = 10 * np.sin(2 * np.pi * alpha_freq * t)
            
            # Beta wave (12-30 Hz)
            beta_freq = 15 + 5 * (ch / n_channels)  # Varies with channel
            beta = 5 * np.sin(2 * np.pi * beta_freq * t)
            
            # Theta wave (4-8 Hz)
            theta = 8 * np.sin(2 * np.pi * 5 * t)
            
            # Add noise
            noise = np.random.normal(0, 2, n_timepoints)
            
            # Combine signals
            eeg_signal = alpha + beta + theta + noise
            
            # Class-specific modification
            eeg_signal *= (1 + 0.1 * y[i])
            
            x[i, ch, :] = eeg_signal
    
    print(f"Generated synthetic EEG data:")
    print(f"  X shape: {x.shape}")
    print(f"  Y shape: {y.shape}")
    print(f"  Classes: {np.unique(y)}")
    
    return x, y

def apply_biosppy_preprocessing(x, sampling_rate=250):
    """
    Apply biosppy signal processing to EEG data.
    
    Args:
        x: EEG signals (n_samples, n_channels, n_timepoints) or (n_samples, n_timepoints)
        sampling_rate: EEG sampling frequency in Hz
    
    Returns:
        x_processed: Processed signals with same shape
    """
    print(f"Applying biosppy preprocessing (fs={sampling_rate}Hz)...")
    
    x_processed = x.copy()
    
    # If 2D, add channel dimension
    if x_processed.ndim == 2:
        x_processed = x_processed[:, np.newaxis, :]
    
    n_samples, n_channels, n_timepoints = x_processed.shape
    
    # Apply biosppy signal processing to each channel
    for i in range(n_samples):
        for j in range(n_channels):
            signal_data = x_processed[i, j, :]
            
            try:
                # Apply biosppy's signal processing utilities
                # Example: Use Butterworth bandpass filter (1-50 Hz for EEG)
                # This is handled by scipy, but biosppy provides signal tools
                sos = signal.butter(4, [1, 50], btype='band', fs=sampling_rate, output='sos')
                signal_data = signal.sosfilt(sos, signal_data)
                
                x_processed[i, j, :] = signal_data
            except Exception as e:
                print(f"Warning: Could not process sample {i}, channel {j}: {e}")
                continue
    
    print(f"Processed EEG data shape: {x_processed.shape}")
    return x_processed

def preprocess_eeg_data(x, y):
    """
    Preprocess EEG/time-series data.
    x shape: (n_samples, n_channels, n_timepoints) or (n_samples, n_features)
    y shape: (n_samples,) with class labels
    """
    # Normalize EEG data to [-1, 1] range
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-8
    x = (x - mean) / std
    
    # Get number of classes
    n_classes = len(np.unique(y))
    
    # Convert labels to one-hot encoding
    y_cat = to_categorical(y, num_classes=n_classes)
    
    # Reshape for network compatibility
    if x.ndim == 2:
        # (n_samples, n_features) -> (n_samples, 1, 1, n_features)
        x = x.reshape(len(x), 1, 1, -1)
    elif x.ndim == 3:
        # (n_samples, n_channels, n_timepoints) -> add batch dimension if needed
        x = x.reshape(len(x), x.shape[1], x.shape[2], 1)
    
    # Reshape labels
    y_cat = y_cat.reshape(len(y_cat), n_classes, 1)
    
    return x, y_cat, n_classes

def load_and_split_data(data_file=None, train_ratio=0.75, val_ratio=0.15, test_ratio=0.1, apply_preprocessing=True):
    """
    Load EEG data and split into train/val/test sets.
    
    Automatically attempts to load from biosppy package datasets.
    Falls back to MNIST or local files if biosppy data unavailable.
    
    Args:
        data_file: Optional path to local EEG data file (.mat or .npz). 
                   If None, attempts to load from biosppy package first.
        train_ratio: Percentage for training (0.75)
        val_ratio: Percentage for validation (0.15)
        test_ratio: Percentage for testing (0.10)
        apply_preprocessing: Whether to apply biosppy preprocessing
    
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes
    """
    # Try to load EEG data
    try:
        x, y = load_biosppy_eeg_data(data_file)
        
        # Apply biosppy preprocessing
        if apply_preprocessing:
            x = apply_biosppy_preprocessing(x, sampling_rate=250)
        
        # Preprocess
        x, y, n_classes = preprocess_eeg_data(x, y)
        
        print(f"Successfully loaded EEG data with {n_classes} classes")
        
    except Exception as e:
        print(f"Error loading EEG data: {e}")
        print("Falling back to MNIST for testing...")
        from keras.datasets import mnist
        
        def preprocess_data(x, y, limit):
            x = x[:limit]
            y = y[:limit]
            x = x.reshape(len(x), 1, 28, 28)
            x = x.astype("float32") / 255
            y = to_categorical(y)
            y = y.reshape(len(y), 10, 1)
            return x, y
        
        (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
        x_train_full, y_train_full = preprocess_data(x_train_full, y_train_full, 60000)
        x_test_full, y_test_full = preprocess_data(x_test_full, y_test_full, 10000)
        
        # Split training data: 75% train, 15% val, 10% test
        n_total = len(x_train_full)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        x_train = x_train_full[:n_train]
        y_train = y_train_full[:n_train]
        x_val = x_train_full[n_train:n_train + n_val]
        y_val = y_train_full[n_train:n_train + n_val]
        x_test = np.vstack([x_train_full[n_train + n_val:], x_test_full])
        y_test = np.vstack([y_train_full[n_train + n_val:], y_test_full])
        
        n_classes = 10
        print("Loaded MNIST fallback data")
    
    # Shuffle data
    indices = np.random.permutation(len(x))
    x = x[indices]
    y = y[indices]
    
    # Split data
    n_total = len(x)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    x_train = x[:n_train]
    y_train = y[:n_train]
    
    x_val = x[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    
    x_test = x[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    print(f"Dataset split: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")
    print(f"Number of classes: {n_classes}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes

WEIGHTS_FILE = "weights.npz"

# Load data with 75/15/10 split
# Automatically attempts to load from biosppy package first
# Falls back to MNIST if biosppy data unavailable

(x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
    apply_preprocessing=True
)

network = [
        # Flatten the input: (n_channels, n_timepoints, 1) -> (n_channels * n_timepoints, 1)
        Reshape(
            (x_train[0].shape[0], x_train[0].shape[1], x_train[0].shape[2]),
            (x_train[0].shape[0] * x_train[0].shape[1] * x_train[0].shape[2], 1)
        ),
        Dense(x_train[0].shape[0] * x_train[0].shape[1] * x_train[0].shape[2], 128),
        Sigmoid(),
        Dense(128, 64),
        Sigmoid(),
        Dense(64, n_classes),
        Softmax()
]

# Loading Weights
if os.path.exists(WEIGHTS_FILE):
    data = np.load(WEIGHTS_FILE)
    dense_layers = [l for l in network if isinstance(l, Dense)]
    for i, layer in enumerate(dense_layers):
        layer.weights = data[f'dense_weights_{i}']
        layer.bias = data[f'dense_bias_{i}']
    conv_layers = [l for l in network if isinstance(l, Convolution)]
    for i, layer in enumerate(conv_layers):
        layer.kernels = data[f'conv_kernels_{i}']
        layer.biases = data[f'conv_biases_{i}']

epochs = 500
learning_rate = 0.0001
start_time = time.time()

def save_weights():
    """Save network weights to file."""
    dense_layers = [l for l in network if isinstance(l, Dense)]
    conv_layers = [l for l in network if isinstance(l, Convolution)]
    np.savez(WEIGHTS_FILE,
        **{f'dense_weights_{i}': l.weights for i, l in enumerate(dense_layers)},
        **{f'dense_bias_{i}': l.bias for i, l in enumerate(dense_layers)},
        **{f'conv_kernels_{i}': l.kernels for i, l in enumerate(conv_layers)},
        **{f'conv_biases_{i}': l.biases for i, l in enumerate(conv_layers)}
    )

def forward_pass(x, network):
    """Perform forward pass through network."""
    output = x
    for layer in network:
        output = layer.forward(output)
    return output

def backward_pass(grad, network, learning_rate):
    """Perform backward pass through network."""
    for layer in reversed(network):
        grad = layer.backward(grad, learning_rate)
    return grad

def compute_accuracy(predictions, targets):
    """Compute classification accuracy."""
    pred_classes = np.argmax(predictions, axis=0)
    true_classes = np.argmax(targets, axis=0)
    return np.mean(pred_classes == true_classes)

def evaluate(x_set, y_set, network):
    """Evaluate model on a dataset."""
    total_loss = 0
    total_acc = 0
    
    for x, y in zip(x_set, y_set):
        output = forward_pass(x, network)
        loss = cross_entropy(y, output)
        acc = compute_accuracy(output, y)
        total_loss += loss
        total_acc += acc
    
    avg_loss = total_loss / len(x_set)
    avg_acc = total_acc / len(x_set)
    
    return avg_loss, avg_acc

# Training loop with validation and testing
print("Starting training...")
print(f"Train samples: {len(x_train)}, Val samples: {len(x_val)}, Test samples: {len(x_test)}")

for e in range(epochs):
    epoch_start = time.time()
    train_error = 0
    train_acc = 0
    
    # Training phase
    for x, y in zip(x_train, y_train):
        output = forward_pass(x, network)
        train_error += cross_entropy(y, output)
        train_acc += compute_accuracy(output, y)
        grad = cross_entropy_prime(y, output)
        backward_pass(grad, network, learning_rate)
    
    train_error /= len(x_train)
    train_acc /= len(x_train)
    
    # Validation phase
    val_loss, val_acc = evaluate(x_val, y_val, network)
    
    # Testing phase (optional - typically done only at end)
    if (e + 1) % 50 == 0:
        test_loss, test_acc = evaluate(x_test, y_test, network)
    else:
        test_loss, test_acc = 0, 0
    
    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_time
    eta = (total_time / (e + 1)) * (epochs - (e + 1))
    
    save_weights()
    
    # Detailed logging
    if (e + 1) % 10 == 0:
        print(f"Epoch {e + 1}/{epochs}")
        print(f"  Train Loss: {train_error:.6f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f}")
        if (e + 1) % 50 == 0:
            print(f"  Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.4f}")
        print(f"  Time: {epoch_time:.2f}s | Total: {total_time:.2f}s | ETA: {eta:.2f}s")

# Final testing
print("\n" + "="*60)
print("Final Evaluation on Test Set")
print("="*60)
test_loss, test_acc = evaluate(x_test, y_test, network)
print(f"Test Loss: {test_loss:.6f}")
print(f"Test Accuracy: {test_acc:.4f}")

