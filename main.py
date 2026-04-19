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

def get_current_dataset_index():
    """
    Get and increment the current dataset index.
    Cycles through 0-31 (32 total datasets from BNCI Horizon 2020).
    """
    counter_file = "current_dataset_index.txt"
    
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            try:
                current_index = int(f.read().strip())
            except:
                current_index = 0
    else:
        current_index = 0
    
    # Save the next index for next run
    next_index = (current_index + 1) % 32
    with open(counter_file, 'w') as f:
        f.write(str(next_index))
    
    return current_index

# BNCI Horizon 2020 Dataset Configurations
BNCI_DATASETS = [
    # Format: (dataset_id, base_url_suffix, file_pattern, description)
    ("001-2014", "001-2014", ["A01T", "A02T", "A03T", "A04T", "A05T", "A06T", "A07T", "A08T", "A09T"], "Four class motor imagery - 9 subjects"),
    ("002-2014", "002-2014", ["S01T", "S02T", "S03T", "S04T", "S05T", "S06T", "S07T", "S08T", "S09T", "S10T", "S11T", "S12T", "S13T", "S14T"], "Two class motor imagery - 14 subjects"),
    ("003-2014", "003-2014", ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"], "Mental arithmetic (fNIRS) - 8 subjects"),
    ("004-2014", "004-2014", ["B01T", "B02T", "B03T", "B04T", "B05T", "B06T", "B07T", "B08T", "B09T"], "Two class motor imagery - 9 subjects"),
    ("005-2014", "005-2014", ["S01", "S02"], "Auditory oddball during hypnosis - 2 subjects"),
    ("006-2014", "006-2014", ["S01", "S02"], "SCP training in stroke - 2 subjects"),
    ("007-2014", "007-2014", ["S00", "S01", "S02", "S04", "S05", "S06", "S07", "S09", "S10"], "Two-finger gameplay - 10 subjects"),
    ("008-2014", "008-2014", ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08"], "P300 speller with ALS - 8 subjects"),
    ("009-2014", "009-2014", ["A01S", "A02S", "A03S", "A04S", "A05S", "A06S", "A07S", "A08S", "A09S", "A10S"], "Covert and overt ERP-based BCI - 10 subjects"),
    ("001-2015", "001-2015", ["S01A", "S02A", "S03A", "S04A", "S05A", "S06A", "S07A", "S08A", "S09A", "S10A", "S11A", "S12A"], "Autocalibration and recurrent adaptation - 12 subjects"),
    ("002-2015", "002-2015", ["S01"], "Neuroprosthetic control - 1 subject"),
    ("003-2015", "003-2015", ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"], "Visual P300 speller - 10 subjects"),
    ("004-2015", "004-2015", ["A", "C", "D", "E", "F", "G", "H", "J", "K"], "Individual imagery - 9 subjects"),
]

def load_bnci_dataset(dataset_index):
    """
    Load a specific BNCI Horizon 2020 dataset.
    
    Args:
        dataset_index: Index into BNCI_DATASETS (0-12 implemented, more can be added)
    
    Returns:
        (x, y) or None if failed
    """
    if dataset_index >= len(BNCI_DATASETS):
        print(f"Dataset index {dataset_index} not yet implemented (total: {len(BNCI_DATASETS)})")
        return None
    
    dataset_id, url_suffix, file_list, description = BNCI_DATASETS[dataset_index]
    
    print(f"\n{'='*60}")
    print(f"Loading BNCI Dataset {dataset_index + 1}/32")
    print(f"ID: {dataset_id}")
    print(f"Description: {description}")
    print(f"{'='*60}\n")
    
    base_url = f"https://bnci-horizon-2020.eu/database/data-sets/{url_suffix}/"
    
    # Try to download and load the first available file from this dataset
    for filename in file_list:
        try:
            dataset_url = base_url + filename + ".mat"
            print(f"Attempting to download: {filename}.mat...")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                mat_path = os.path.join(tmpdir, f"{filename}.mat")
                urllib.request.urlretrieve(dataset_url, mat_path)
                
                data = loadmat(mat_path)
                
                # Try different key patterns
                x, y = None, None
                if 'X' in data and 'y' in data:
                    x = data['X']
                    y = data['y'].flatten()
                elif 'x' in data and 'y' in data:
                    x = data['x']
                    y = data['y'].flatten()
                elif 'signal' in data and 'label' in data:
                    x = data['signal']
                    y = data['label'].flatten()
                
                if x is not None and y is not None:
                    print(f"✓ Successfully loaded {filename}.mat")
                    print(f"  Shape: X={x.shape}, Y={y.shape}, Classes={len(np.unique(y))}")
                    return x, y
                else:
                    available_keys = [k for k in data.keys() if not k.startswith('__')]
                    print(f"  Keys in file: {available_keys}")
        
        except urllib.error.HTTPError as e:
            print(f"  File not found: {e.code}")
            continue
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue
    
    print(f"Could not load any files from dataset {dataset_id}")
    return None

def load_biosppy_eeg_data(data_file=None):
    """
    Load EEG data from available sources, cycling through BNCI Horizon 2020 datasets.
    
    Attempts to (in order):
    1. Cycle through BNCI Horizon 2020 datasets (13 implemented)
    2. Load from local data file (.mat or .npz)
    3. Generate synthetic EEG data using biosppy utilities
    4. Fall back to MNIST
    
    Args:
        data_file: Optional path to local data file (.mat or .npz). 
    
    Returns:
        x: EEG signals (n_samples, n_channels, n_timepoints)
        y: Class labels (n_samples,)
    """
    
    # Try to load from cycling BNCI Horizon 2020 datasets
    current_dataset_idx = get_current_dataset_index()
    bnci_data = load_bnci_dataset(current_dataset_idx)
    if bnci_data is not None:
        x, y = bnci_data
        print(f"Loaded BNCI Horizon 2020 data")
        print(f"X shape: {x.shape}, Y shape: {y.shape}")
        return x, y
    
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
    try:
        data = np.load(WEIGHTS_FILE)
        dense_layers = [l for l in network if isinstance(l, Dense)]
        for i, layer in enumerate(dense_layers):
            if f'dense_weights_{i}' in data and f'dense_bias_{i}' in data:
                weights = data[f'dense_weights_{i}']
                bias = data[f'dense_bias_{i}']
                # Only load if shapes match
                if weights.shape == layer.weights.shape and bias.shape == layer.bias.shape:
                    layer.weights = weights
                    layer.bias = bias
                else:
                    print(f"Weight shape mismatch for dense layer {i}, skipping")
        conv_layers = [l for l in network if isinstance(l, Convolution)]
        for i, layer in enumerate(conv_layers):
            if f'conv_kernels_{i}' in data and f'conv_biases_{i}' in data:
                kernels = data[f'conv_kernels_{i}']
                biases = data[f'conv_biases_{i}']
                # Only load if shapes match
                if kernels.shape == layer.kernels.shape and biases.shape == layer.biases.shape:
                    layer.kernels = kernels
                    layer.biases = biases
                else:
                    print(f"Weight shape mismatch for conv layer {i}, skipping")
        print("Loaded compatible weights")
    except Exception as e:
        print(f"Could not load weights: {e}")
        print("Starting with fresh weights")

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

def run_training_loop(num_runs=5, epochs_per_run=50):
    """
    Run multiple training sessions, cycling through BNCI datasets.
    
    Args:
        num_runs: Number of training sessions to run
        epochs_per_run: Number of epochs per training session
    """
    print(f"\n{'='*80}")
    print(f"Starting BNCI Dataset Training Loop")
    print(f"Will run {num_runs} training sessions, {epochs_per_run} epochs each")
    print(f"Cycling through BNCI Horizon 2020 datasets")
    print(f"{'='*80}\n")
    
    for run in range(num_runs):
        print(f"\n{'#'*60}")
        print(f"TRAINING SESSION {run + 1}/{num_runs}")
        print(f"{'#'*60}\n")
        
        try:
            # Load data (will automatically cycle to next BNCI dataset)
            (x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
                apply_preprocessing=True
            )
            
            # Build network dynamically based on data shape
            network = [
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
            
            # Load existing weights if compatible
            if os.path.exists(WEIGHTS_FILE):
                try:
                    data = np.load(WEIGHTS_FILE)
                    dense_layers = [l for l in network if isinstance(l, Dense)]
                    for i, layer in enumerate(dense_layers):
                        if f'dense_weights_{i}' in data and f'dense_bias_{i}' in data:
                            weights = data[f'dense_weights_{i}']
                            bias = data[f'dense_bias_{i}']
                            if weights.shape == layer.weights.shape and bias.shape == layer.bias.shape:
                                layer.weights = weights
                                layer.bias = bias
                            else:
                                print(f"Weight shape mismatch for dense layer {i}, skipping")
                    print("Loaded compatible weights")
                except Exception as e:
                    print(f"Could not load weights: {e}")
                    print("Starting with fresh weights")
            
            # Training loop for this session
            print(f"Training on {len(x_train)} samples, {n_classes} classes")
            print(f"Network: {len(network)} layers")
            
            for epoch in range(epochs_per_run):
                epoch_start = time.time()
                
                # Train on all samples
                total_loss = 0
                total_acc = 0
                
                for x, y in zip(x_train, y_train):
                    output = forward_pass(x, network)
                    loss = cross_entropy(y, output)
                    acc = compute_accuracy(output, y)
                    total_loss += loss
                    total_acc += acc
                    
                    grad = cross_entropy_prime(y, output)
                    backward_pass(grad, network, learning_rate)
                
                train_loss = total_loss / len(x_train)
                train_acc = total_acc / len(x_train)
                
                # Validate
                val_loss, val_acc = evaluate(x_val, y_val, network)
                
                # Test every 10 epochs
                if (epoch + 1) % 10 == 0:
                    test_loss, test_acc = evaluate(x_test, y_test, network)
                    print(f"Epoch {epoch+1:3d}/{epochs_per_run} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f} | Test: {test_loss:.4f}/{test_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1:3d}/{epochs_per_run} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f}")
            
            # Save weights after this session
            save_weights()
            print(f"✓ Session {run + 1} complete - weights saved")
            
        except Exception as e:
            print(f"❌ Error in session {run + 1}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Training loop complete! Ran {num_runs} sessions")
    print(f"{'='*80}")
    
    # Cleanup temporary files
    cleanup_files()

def cleanup_files():
    """Clean up temporary files created during training."""
    files_to_remove = [
        "current_dataset_index.txt"  # Keep weights.npz for accumulation across runs
    ]
    
    print("\n🧹 Cleaning up temporary files...")
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  ✓ Removed {file_path}")
            except Exception as e:
                print(f"  ❌ Could not remove {file_path}: {e}")
        else:
            print(f"  - {file_path} not found")
    
    print("✓ Cleanup complete")

if __name__ == "__main__":
    # Run the training loop for all 32 BNCI datasets
    run_training_loop(num_runs=32, epochs_per_run=2014)

