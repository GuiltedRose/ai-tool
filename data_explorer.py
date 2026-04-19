"""
EEG Data Exploration and Preprocessing Utilities
Designed for BNCI/local biometric datasets in BCI projects.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def load_and_explore_dataset():
    """
    Load the project's configured BCI dataset and print comprehensive information.
    """
    try:
        from main import load_biosppy_eeg_data
        x, y = load_biosppy_eeg_data(dataset_index=0)
    except Exception as e:
        print(f"Error loading BCI dataset: {e}")
        return None, None
    
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")
    print(f"X dtype: {x.dtype}")
    print(f"Y dtype: {y.dtype}")
    print(f"X range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Y unique classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Missing values in X: {np.isnan(x).sum()}")
    print(f"Missing values in Y: {np.isnan(y).sum()}")
    print("=" * 60)
    
    return x, y

def preprocess_eeg_with_filtering(x, y, lowcut=1.0, highcut=50.0, order=4, fs=250):
    """
    Preprocess EEG data with bandpass filtering.
    
    Args:
        x: EEG data (n_samples, n_features) or (n_samples, n_channels, n_timepoints)
        y: Labels
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order
        fs: Sampling frequency (Hz)
    
    Returns:
        x_filtered: Filtered and normalized EEG data
        y: Original labels
    """
    print(f"\nApplying bandpass filter ({lowcut}-{highcut} Hz, order={order})...")
    
    x_orig_shape = x.shape

    if x.ndim == 2 and x.shape[1] < 32:
        print(
            "Skipping bandpass filter for 2D channel-feature data "
            f"with shape {x.shape}; not enough timepoints per sample."
        )
        return x, y
    
    # Flatten to 2D if needed for filtering
    if x.ndim == 3:
        n_samples, n_channels, n_timepoints = x.shape
        x = x.reshape(n_samples * n_channels, n_timepoints)
    
    # Design filter
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if low <= 0 or high >= 1:
        print(f"Warning: Filter frequencies out of range for fs={fs}Hz")
        b, a = signal.butter(order, [low + 0.01, high - 0.01], btype='band')
    else:
        b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter to each channel
    x_filtered = np.zeros_like(x)
    for i in range(x.shape[0]):
        x_filtered[i] = signal.filtfilt(b, a, x[i])
    
    # Reshape back
    if len(x_orig_shape) == 3:
        x_filtered = x_filtered.reshape(x_orig_shape)
    
    # Normalize
    mean = np.mean(x_filtered, axis=-1, keepdims=True)
    std = np.std(x_filtered, axis=-1, keepdims=True) + 1e-8
    x_filtered = (x_filtered - mean) / std
    
    print(f"Filtered data shape: {x_filtered.shape}")
    print(f"Filtered data range: [{x_filtered.min():.4f}, {x_filtered.max():.4f}]")
    
    return x_filtered, y

def extract_features(x, y, method='statistical'):
    """
    Extract features from EEG data.
    
    Args:
        x: EEG data (n_samples, n_channels, n_timepoints)
        y: Labels
        method: 'statistical', 'frequency', or 'time-frequency'
    
    Returns:
        x_features: Feature matrix (n_samples, n_features)
        y: Original labels
    """
    print(f"\nExtracting {method} features...")
    
    if x.ndim == 2:
        # (n_samples, n_features) - already feature format
        print("Data is already in feature format")
        return x, y
    
    if method == 'statistical':
        # Extract: mean, std, min, max, range, skew, kurtosis
        n_samples = x.shape[0]
        features = []
        
        for sample in x:
            stats = []
            for channel in sample:
                stats.extend([
                    np.mean(channel),
                    np.std(channel),
                    np.min(channel),
                    np.max(channel),
                    np.max(channel) - np.min(channel),
                    np.mean(np.abs(np.diff(channel)))  # Mean absolute difference
                ])
            features.append(stats)
        
        x_features = np.array(features)
        print(f"Extracted {x_features.shape[1]} statistical features")
        
    elif method == 'frequency':
        # Extract frequency domain features (PSD)
        from scipy.signal import welch
        
        n_samples = x.shape[0]
        n_channels = x.shape[1]
        features = []
        
        for sample in x:
            psd_features = []
            for channel in sample:
                freqs, psd = welch(channel, fs=250, nperseg=256)
                # Extract PSD in frequency bands
                psd_features.extend([
                    np.mean(psd[freqs < 4]),      # Delta
                    np.mean(psd[(freqs >= 4) & (freqs < 8)]),   # Theta
                    np.mean(psd[(freqs >= 8) & (freqs < 12)]),  # Alpha
                    np.mean(psd[(freqs >= 12) & (freqs < 30)]), # Beta
                    np.mean(psd[freqs >= 30]),     # Gamma
                ])
            features.append(psd_features)
        
        x_features = np.array(features)
        print(f"Extracted {x_features.shape[1]} frequency domain features")
    
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")
    
    return x_features, y

def visualize_sample(x, y, sample_idx=0, num_channels=5):
    """
    Visualize a sample from the dataset.
    
    Args:
        x: EEG data
        y: Labels
        sample_idx: Index of sample to visualize
        num_channels: Number of channels to plot (if available)
    """
    print(f"\nVisualizing sample {sample_idx} (class {y[sample_idx]})...")
    
    sample = x[sample_idx]
    
    if sample.ndim == 1:
        # Single channel
        plt.figure(figsize=(12, 4))
        plt.plot(sample)
        plt.title(f"EEG Signal - Class {y[sample_idx]}")
        plt.xlabel("Time point")
        plt.ylabel("Amplitude")
        plt.show()
    
    elif sample.ndim == 2:
        # Multiple channels
        n_channels = min(sample.shape[0], num_channels)
        plt.figure(figsize=(12, n_channels * 2))
        
        for i in range(n_channels):
            plt.subplot(n_channels, 1, i + 1)
            plt.plot(sample[i])
            plt.ylabel(f"Ch {i}")
        
        plt.suptitle(f"EEG Signal - Class {y[sample_idx]}")
        plt.xlabel("Time point")
        plt.tight_layout()
        plt.show()

def check_dataset_balance(y):
    """Check and visualize class balance."""
    print("\nDataset Balance:")
    unique_classes, counts = np.unique(y, return_counts=True)
    
    for cls, count in zip(unique_classes, counts):
        percentage = (count / len(y)) * 100
        print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # Check if balanced
    mean_count = np.mean(counts)
    imbalance = np.std(counts) / mean_count
    
    if imbalance > 0.1:
        print(f"\nWarning: Dataset is imbalanced (std/mean = {imbalance:.2f})")
        print("Consider using stratified splitting in train/val/test")

if __name__ == "__main__":
    print("EEG Dataset Exploration Tool")
    print("=" * 60)
    
    # Load and explore
    x, y = load_and_explore_dataset()
    
    if x is not None:
        # Check balance
        check_dataset_balance(y)
        
        # Visualize a sample
        visualize_sample(x, y, sample_idx=0)
        
        # Test preprocessing
        x_filtered, _ = preprocess_eeg_with_filtering(x, y)
        
        # Test feature extraction
        x_features, _ = extract_features(x, y, method='statistical')
