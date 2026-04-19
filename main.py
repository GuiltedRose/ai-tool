import os
import numpy as np
from scipy import signal
from scipy.io import loadmat, savemat
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
        tmp = np.exp(input - np.max(input))
        self.output = tmp / np.sum(tmp)
        return self.output
    def backward(self, output_gradient, learning_rate):
       return output_gradient

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
    
    def unpack_matlab_cell(data):
        """Recursively unpack MATLAB object cells and scalar arrays."""
        while isinstance(data, np.ndarray) and data.dtype == object and data.shape == (1, 1):
            data = data[0, 0]
        return data

    def first_matching_field(field_names, candidates):
        for candidate in candidates:
            for field in field_names:
                if field.lower() == candidate:
                    return field
        return None

    def select_struct_fields(trial_data):
        field_names = trial_data.dtype.names
        print(f"  Structured fields: {field_names}")

        all_fields = {}
        sizes = {}
        for field in field_names:
            field_data = trial_data[field]
            all_fields[field] = field_data
            sizes[field] = field_data.size if hasattr(field_data, 'size') else 1
            if hasattr(field_data, 'shape'):
                print(f"    {field}: shape={field_data.shape}, dtype={field_data.dtype}")

        x_field = first_matching_field(field_names, ['x', 'signal', 'signals', 'eeg', 'data'])
        y_field = first_matching_field(field_names, ['y', 'label', 'labels', 'target', 'targets', 'y_true'])
        trial_field = first_matching_field(field_names, ['trial', 'trials', 'events', 'event'])

        if x_field is None:
            excluded_x_fields = {'trial', 'trials', 'classes', 'classes_stim', 'channels', 'gender', 'age', 'alsfrs', 'onsetals', 'fs'}
            x_candidates = [f for f in field_names if f.lower() not in excluded_x_fields]
            x_field = max(x_candidates or field_names, key=lambda f: sizes[f])
            print(f"  Inferred X field: {x_field}")
        else:
            print(f"  Selected X field: {x_field}")

        if y_field is None:
            excluded_y_fields = {'trial', 'trials', 'classes', 'classes_stim', 'channels', 'gender', 'age', 'alsfrs', 'onsetals', 'fs'}
            remaining = [f for f in field_names if f != x_field and f.lower() not in excluded_y_fields]
            if remaining:
                target_size = all_fields[x_field].shape[0] if hasattr(all_fields[x_field], 'shape') else None
                same_length = [
                    f for f in remaining
                    if hasattr(all_fields[f], 'shape') and all_fields[f].shape and all_fields[f].shape[0] == target_size
                ]
                y_field = min(same_length or remaining, key=lambda f: sizes[f])
                print(f"  Inferred y field: {y_field}")
        else:
            print(f"  Selected y field: {y_field}")

        x_raw = unpack_matlab_cell(all_fields[x_field]) if x_field else None
        y_raw = unpack_matlab_cell(all_fields[y_field]) if y_field else None
        trial_raw = unpack_matlab_cell(all_fields[trial_field]) if trial_field else None
        return x_raw, y_raw, trial_raw

    def epoch_continuous_signal(x_raw, y_raw, trial_raw):
        """
        Align BNCI continuous streams with trial-level labels.

        Returns either:
        - epochs shaped (trials, channels, timepoints), labels shaped (trials,)
        - row samples shaped (samples, features), labels shaped (samples,)
        - None if the block has no usable labels.
        """
        if x_raw is None or y_raw is None:
            return None

        x = np.asarray(x_raw)
        y = np.asarray(y_raw).squeeze()

        if y.size == 0:
            print("  Skipping block with empty labels")
            return None

        y = y.reshape(-1)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if x.ndim >= 2 and x.shape[0] == len(y):
            print(f"  Using sample-aligned block: X={x.shape}, y={y.shape}")
            return x, y

        if trial_raw is None:
            print(f"  Skipping unaligned block: X={x.shape}, y={y.shape}, no trial markers")
            return None

        trial = np.asarray(trial_raw).squeeze().reshape(-1)
        if trial.size == 0:
            print(f"  Skipping unaligned block: X={x.shape}, y={y.shape}, empty trial markers")
            return None

        trial = trial.astype(np.int64)
        trial = trial[trial > 0]
        if trial.size == 0:
            print(f"  Skipping block with no positive trial markers")
            return None

        starts = trial - 1 if np.min(trial) >= 1 else trial
        n_trials = min(len(starts), len(y))
        starts = starts[:n_trials]
        y = y[:n_trials]

        valid_starts = starts[(starts >= 0) & (starts < x.shape[0])]
        if len(valid_starts) != len(starts):
            keep = (starts >= 0) & (starts < x.shape[0])
            starts = starts[keep]
            y = y[keep]

        if len(starts) == 0:
            print("  Skipping block with no in-range trial starts")
            return None

        if len(starts) > 1:
            epoch_len = int(np.median(np.diff(np.sort(starts))))
        else:
            epoch_len = x.shape[0] - starts[0]

        epoch_len = max(1, min(epoch_len, x.shape[0] - int(np.max(starts))))
        epochs = []
        labels = []
        for start, label in zip(starts, y):
            stop = int(start) + epoch_len
            if stop <= x.shape[0]:
                epochs.append(x[int(start):stop].T)
                labels.append(label)

        if not epochs:
            print("  Skipping block because no full epochs could be cut")
            return None

        epochs = np.asarray(epochs)
        labels = np.asarray(labels)
        print(f"  Epoched continuous block: X={epochs.shape}, y={labels.shape}, epoch_len={epoch_len}")
        return epochs, labels

    def concatenate_chunks(x_chunks, y_chunks):
        if not x_chunks:
            return None

        dims = {x.ndim for x in x_chunks}
        if len(dims) != 1:
            print(f"  Mixed sample ranks found {dims}; flattening chunks before concatenation")
            x_chunks = [x.reshape(x.shape[0], -1) for x in x_chunks]

        if x_chunks[0].ndim == 3:
            min_channels = min(x.shape[1] for x in x_chunks)
            min_timepoints = min(x.shape[2] for x in x_chunks)
            x_chunks = [x[:, :min_channels, :min_timepoints] for x in x_chunks]
        elif x_chunks[0].ndim == 2:
            min_features = min(x.shape[1] for x in x_chunks)
            x_chunks = [x[:, :min_features] for x in x_chunks]

        return np.concatenate(x_chunks, axis=0), np.concatenate(y_chunks, axis=0)

    x_chunks = []
    y_chunks = []

    # Download and load every file configured for this dataset set.
    for filename in file_list:
        try:
            dataset_url = base_url + filename + ".mat"
            print(f"Attempting to download: {filename}.mat...")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                mat_path = os.path.join(tmpdir, f"{filename}.mat")
                urllib.request.urlretrieve(dataset_url, mat_path)
                
                data = loadmat(mat_path)
                available_keys = [k for k in data.keys() if not k.startswith('__')]
                print(f"  Keys in file: {available_keys}")
                
                if 'data' in data:
                    raw_data = data['data']
                    print(f"  'data' type: {type(raw_data).__name__}, shape: {raw_data.shape}, dtype: {raw_data.dtype}")

                    for cell_index, cell in enumerate(raw_data.flat):
                        trial_data = unpack_matlab_cell(cell)
                        if not isinstance(trial_data, np.ndarray):
                            continue

                        print(f"  Cell {cell_index + 1}/{raw_data.size}: shape={trial_data.shape}, dtype={trial_data.dtype}")

                        if trial_data.dtype.names:
                            x_raw, y_raw, trial_raw = select_struct_fields(trial_data)
                            print(f"  Unpacked X type: {type(x_raw)}, ", end="")
                            if isinstance(x_raw, np.ndarray):
                                print(f"shape: {x_raw.shape}, dtype: {x_raw.dtype}")
                            else:
                                print(f"length: {len(x_raw) if hasattr(x_raw, '__len__') else 'N/A'}")

                            print(f"  Unpacked y type: {type(y_raw)}, ", end="")
                            if isinstance(y_raw, np.ndarray):
                                print(f"shape: {y_raw.shape}, dtype: {y_raw.dtype}")
                            else:
                                print(f"value: {y_raw}")

                            prepared = epoch_continuous_signal(x_raw, y_raw, trial_raw)
                            if prepared is not None:
                                x_prepared, y_prepared = prepared
                                x_chunks.append(x_prepared)
                                y_chunks.append(y_prepared)
                        else:
                            print(f"  Skipping unstructured cell: shape={trial_data.shape}")
                else:
                    print(f"  No 'data' key found, skipping")

        
        except urllib.error.HTTPError as e:
            print(f"  File not found: {e.code}")
            continue
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    combined = concatenate_chunks(x_chunks, y_chunks)
    if combined is not None:
        x_all, y_all = combined
        print(f"✓ Loaded all usable files for dataset {dataset_id}")
        print(f"  Combined X shape: {x_all.shape}")
        print(f"  Combined y shape: {y_all.shape}")
        return x_all, y_all

    print(f"Could not load any usable samples from dataset {dataset_id}")
    return None

def load_biosppy_eeg_data(data_file=None, dataset_index=0, require_bnci=False):
    """
    Load EEG data from available sources, cycling through BNCI Horizon 2020 datasets.
    
    Attempts to (in order):
    1. Load the requested BNCI Horizon 2020 dataset
    2. Load from local data file (.mat or .npz)
    3. Generate synthetic EEG data using biosppy utilities
    
    Args:
        data_file: Optional path to local data file (.mat or .npz). 
        dataset_index: Zero-based BNCI dataset index to load when using BNCI.
        require_bnci: If True, raise when the requested BNCI dataset cannot load.
    
    Returns:
        x: EEG signals (n_samples, n_channels, n_timepoints)
        y: Class labels (n_samples,)
    """
    
    # Try to load the requested BNCI Horizon 2020 dataset.
    bnci_data = load_bnci_dataset(dataset_index)
    if bnci_data is not None:
        x, y = bnci_data
        print(f"Loaded BNCI Horizon 2020 data")
        print(f"X shape: {x.shape}, Y shape: {y.shape}")
        return x, y

    if require_bnci:
        raise FileNotFoundError(f"BNCI dataset {dataset_index + 1}/32 could not be loaded")
    
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

    if x_processed.ndim == 2 and x_processed.shape[1] < 32:
        print(
            "Skipping bandpass preprocessing for 2D channel-feature data "
            f"with shape {x_processed.shape}; not enough timepoints per sample."
        )
        return x_processed
    
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

def encode_labels(y):
    """Map arbitrary class labels to stable zero-based one-hot vectors."""
    y = np.squeeze(np.array(y))
    y = y.flatten()
    if y.size == 0:
        raise ValueError("No labels found in data")

    labels = []
    for label in y:
        if isinstance(label, np.ndarray):
            labels.append(tuple(np.squeeze(label).tolist()))
        else:
            labels.append(label)

    unique_classes = np.array(sorted(set(labels), key=lambda value: str(value)), dtype=object)
    class_to_index = {label: idx for idx, label in enumerate(unique_classes)}
    y_index = np.array([class_to_index[label] for label in labels], dtype=np.int32)
    y_cat = np.eye(len(unique_classes), dtype=np.float32)[y_index]
    return y_cat.reshape(len(y_cat), len(unique_classes), 1), len(unique_classes)

def normalize_biometric_data(x):
    """
    Z-score biometric signals without mixing unrelated channel statistics.
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3:
        # Epoch data: normalize each channel across samples and time.
        mean = np.mean(x, axis=(0, 2), keepdims=True)
        std = np.std(x, axis=(0, 2), keepdims=True) + 1e-8
    elif x.ndim == 2:
        # Feature/channel rows: normalize each feature/channel across samples.
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True) + 1e-8
    else:
        mean = np.mean(x, keepdims=True)
        std = np.std(x, keepdims=True) + 1e-8

    return (x - mean) / std

def reshape_eeg_for_network(x):
    """
    Convert biometric samples to the channel-first shape expected by the
    custom layers: (n_samples, depth, height, width).
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.squeeze(x)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if x.ndim == 2:
        # Tabular/features or single-channel time series:
        # (samples, features) -> (samples, 1, 1, features)
        return x.reshape(x.shape[0], 1, 1, x.shape[1])

    if x.ndim == 3:
        # EEG/biometric epochs:
        # (samples, channels, timepoints) -> (samples, 1, channels, timepoints)
        return x[:, np.newaxis, :, :]

    if x.ndim == 4:
        # Already channel-first if the singleton/model depth is in axis 1.
        if x.shape[1] <= x.shape[-1]:
            return x
        # Common channels-last layout:
        # (samples, height, width, depth) -> (samples, depth, height, width)
        return np.transpose(x, (0, 3, 1, 2))

    raise ValueError(f"Unsupported EEG data shape: {x.shape}")

def preprocess_eeg_data(x, y):
    """
    Preprocess EEG/time-series data.
    x shape: (n_samples, n_channels, n_timepoints) or (n_samples, n_features)
    y shape: (n_samples,) with class labels
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.array(y).flatten()
    
    # Validate data
    if x.shape[0] != y.shape[0]:
        print(f"Data mismatch: x.shape[0]={x.shape[0]}, y.shape[0]={y.shape[0]}")
        # Truncate to match
        min_samples = min(x.shape[0], y.shape[0])
        x = x[:min_samples]
        y = y[:min_samples]
    
    if x.shape[0] == 0:
        raise ValueError("No valid samples in data")
    
    # Normalize biometric data while preserving channel-specific statistics.
    try:
        x = normalize_biometric_data(x)
    except Exception as e:
        print(f"Warning: Normalization failed: {e}, skipping")
    
    y_cat, n_classes = encode_labels(y)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes for classification, found {n_classes}")

    x = reshape_eeg_for_network(x)
    print(f"Network input shape: {x.shape} (samples, depth, height, width)")
    print(f"Encoded labels shape: {y_cat.shape}; classes: {n_classes}")
    
    return x, y_cat, n_classes

def load_and_split_data(data_file=None, train_ratio=0.75, val_ratio=0.15, test_ratio=0.1, apply_preprocessing=True, dataset_index=0, require_bnci=False):
    """
    Load EEG data and split into train/val/test sets.
    
    Loads BCI/biometric data from BNCI or a local EEG file, then splits it.
    
    Args:
        data_file: Optional path to local EEG data file (.mat or .npz). 
                   If None, attempts to load from biosppy package first.
        train_ratio: Percentage for training (0.75)
        val_ratio: Percentage for validation (0.15)
        test_ratio: Percentage for testing (0.10)
        apply_preprocessing: Whether to apply biosppy preprocessing
        dataset_index: Zero-based BNCI dataset index to load.
        require_bnci: If True, do not substitute local/synthetic data for BNCI.
    
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes
    """
    x, y = load_biosppy_eeg_data(data_file, dataset_index=dataset_index, require_bnci=require_bnci)

    # Apply biosppy preprocessing
    if apply_preprocessing:
        x = apply_biosppy_preprocessing(x, sampling_rate=250)

    # Preprocess
    x, y, n_classes = preprocess_eeg_data(x, y)

    print(f"Successfully loaded EEG data with {n_classes} classes")
    
    # Shuffle data
    indices = np.random.permutation(len(x))
    x = x[indices]
    y = y[indices]
    
    # Split data
    n_total = len(x)
    if n_total < 3:
        raise ValueError(f"Need at least 3 samples to create train/val/test splits, found {n_total}")

    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_val = 1
        n_train = n_total - 2
    
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

def build_network(input_shape, n_classes):
    """Build a dense classifier for pre-shaped biometric samples."""
    flattened_size = int(np.prod(input_shape))
    return [
        Reshape(input_shape, (flattened_size, 1)),
        Dense(flattened_size, 128),
        Sigmoid(),
        Dense(128, 64),
        Sigmoid(),
        Dense(64, n_classes),
        Softmax()
    ]

def save_weights(network):
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

def compute_class_weights(y_set):
    """Compute inverse-frequency weights for imbalanced BCI classes."""
    class_indices = np.argmax(y_set, axis=1).flatten()
    n_classes = y_set.shape[1]
    counts = np.bincount(class_indices, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(class_indices) / (n_classes * counts)
    return weights.reshape(n_classes, 1)

def sample_weight_for_label(y, class_weights):
    """Return the class weight for a one-hot column label."""
    class_index = int(np.argmax(y))
    return float(class_weights[class_index, 0])

def compute_accuracy(predictions, targets):
    """Compute classification accuracy."""
    pred_classes = np.argmax(predictions, axis=0)
    true_classes = np.argmax(targets, axis=0)
    return np.mean(pred_classes == true_classes)

def evaluate(x_set, y_set, network, class_weights=None):
    """Evaluate model on a dataset."""
    total_loss = 0
    total_acc = 0
    
    for x, y in zip(x_set, y_set):
        output = forward_pass(x, network)
        sample_weight = sample_weight_for_label(y, class_weights) if class_weights is not None else 1.0
        loss = cross_entropy(y, output, sample_weight=sample_weight)
        acc = compute_accuracy(output, y)
        total_loss += loss
        total_acc += acc
    
    avg_loss = total_loss / len(x_set)
    avg_acc = total_acc / len(x_set)
    
    return avg_loss, avg_acc

def run_training_loop(num_runs=32, epochs_per_run=None, learning_rate=0.0001):
    """
    Run one deterministic sweep over BNCI dataset sets.

    One run is one BNCI dataset set: 1/32 of the full configured dataset
    structure. One epoch is one data item from that set's training split.
    By default, each set runs len(x_train) epochs so every training item in
    that set is used once.
    
    Args:
        num_runs: Number of BNCI dataset sets to train on.
        epochs_per_run: Training items to consume per dataset set. If None,
            use every item in the set's training split once.
        learning_rate: Learning rate for training
    """
    print(f"\n{'='*80}")
    print(f"Starting BNCI Dataset Training Loop")
    epoch_plan = "all training items in each set" if epochs_per_run is None else f"{epochs_per_run} items per set"
    print(f"Will run {num_runs} dataset sets, {epoch_plan}")
    print(f"Each dataset set is 1/32 of the BNCI structure")
    print(f"{'='*80}\n")
    
    for run in range(num_runs):
        print(f"\n{'#'*60}")
        print(f"TRAINING SESSION {run + 1}/{num_runs}")
        print(f"{'#'*60}\n")
        
        try:
            # Load a deterministic BNCI dataset: run 1 starts at dataset 1.
            (x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes = load_and_split_data(
                apply_preprocessing=True,
                dataset_index=run,
                require_bnci=True
            )
            
            # Build network dynamically based on the prepared sample shape.
            input_shape = x_train[0].shape
            network = build_network(input_shape, n_classes)
            
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
            class_weights = compute_class_weights(y_train)
            print(f"Training on {len(x_train)} samples, {n_classes} classes")
            print(f"Input sample shape: {input_shape}; flattened size: {int(np.prod(input_shape))}")
            print(f"Class weights: {class_weights.flatten()}")
            print(f"Network: {len(network)} layers")

            total_epochs = len(x_train) if epochs_per_run is None else epochs_per_run
            log_interval = max(1, min(1000, total_epochs))
            print(f"Epochs for this set: {total_epochs} (one training item per epoch)")
            
            total_loss = 0
            total_acc = 0
            
            for epoch in range(total_epochs):
                epoch_start = time.time()

                sample_index = epoch % len(x_train)
                x = x_train[sample_index]
                y = y_train[sample_index]

                output = forward_pass(x, network)
                sample_weight = sample_weight_for_label(y, class_weights)
                loss = cross_entropy(y, output, sample_weight=sample_weight)
                acc = compute_accuracy(output, y)
                total_loss += loss
                total_acc += acc

                grad = cross_entropy_prime(y, output, sample_weight=sample_weight)
                backward_pass(grad, network, learning_rate)

                train_loss = total_loss / (epoch + 1)
                train_acc = total_acc / (epoch + 1)

                should_evaluate = (epoch + 1) == total_epochs or (epoch + 1) % log_interval == 0
                if should_evaluate:
                    val_loss, val_acc = evaluate(x_val, y_val, network, class_weights=class_weights)
                    test_loss, test_acc = evaluate(x_test, y_test, network, class_weights=class_weights)
                    print(f"Epoch {epoch+1:3d}/{total_epochs} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f} | Test: {test_loss:.4f}/{test_acc:.4f}")
            
            # Save weights after this session
            save_weights(network)
            print(f"✓ Session {run + 1} complete - weights saved")
            
        except Exception as e:
            print(f"❌ Error in session {run + 1}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Training loop complete! Ran {num_runs} dataset sets")
    print(f"{'='*80}")
    
    # Cleanup temporary files
    cleanup_files()

def cleanup_files():
    """Clean up temporary files created during training."""
    files_to_remove = []
    
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
    # Run one training epoch per data item in each of the 32 BNCI dataset sets.
    run_training_loop(num_runs=32, epochs_per_run=None, learning_rate=0.0001)
