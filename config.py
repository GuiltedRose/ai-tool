"""
Configuration file for BCI neural network training.
Modify these parameters to adjust training behavior.
"""

# Data Configuration
DATA_CONFIG = {
    'train_ratio': 0.75,      # Percentage of data for training
    'val_ratio': 0.15,        # Percentage of data for validation
    'test_ratio': 0.10,       # Percentage of data for testing
    'random_seed': 42,        # For reproducible splits
    'use_pyscppy': True,      # Use pyscppy; False = MNIST fallback
}

# Preprocessing Configuration
PREPROCESS_CONFIG = {
    'normalize': True,         # Z-score normalization
    'lowcut': 1.0,            # Bandpass filter low cutoff (Hz)
    'highcut': 50.0,          # Bandpass filter high cutoff (Hz)
    'filter_order': 4,        # Butterworth filter order
    'sampling_freq': 250,     # EEG sampling frequency (Hz)
    'apply_filter': False,    # Whether to apply bandpass filter
}

# Network Architecture Configuration
# NOTE: Adjust input_shape and layer sizes based on your EEG data!
NETWORK_CONFIG = {
    'input_shape': (1, 1, 300),    # (depth, height, width) - adjust for your data!
    'conv_kernel_size': 3,          # Convolution kernel size
    'conv_filters': 5,              # Number of convolution filters
    'dense_hidden_1': 100,          # First hidden dense layer size
    'use_dropout': False,           # Add dropout layers (not yet implemented)
    'dropout_rate': 0.5,            # Dropout rate if enabled
    'activation_function': 'sigmoid', # 'sigmoid' or 'tanh'
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 500,                  # Number of training epochs
    'learning_rate': 0.0001,        # Learning rate for SGD
    'batch_size': 1,                # Batch size (1 = SGD)
    'save_interval': 1,             # Save weights every N epochs
    'log_interval': 10,             # Print logs every N epochs
    'test_interval': 50,            # Evaluate on test set every N epochs
    'early_stopping_patience': 50,  # Epochs to wait before stopping if no improvement
    'early_stopping_enabled': False,# Enable early stopping
}

# Output Configuration
OUTPUT_CONFIG = {
    'weights_file': 'weights.npz',          # File to save/load weights
    'log_file': 'training_log.txt',         # File to save training log
    'save_logs': True,                      # Save training logs to file
    'verbose': True,                        # Print progress to console
    'plot_results': False,                  # Plot training curves (requires matplotlib)
    'results_plot_file': 'training_curves.png',
}

# Data Augmentation Configuration (for future implementation)
AUGMENTATION_CONFIG = {
    'enabled': False,
    'noise_std': 0.01,              # Gaussian noise standard deviation
    'time_shift': 0,                # Maximum time shift in samples
    'scaling_factor': 0.1,          # Random scaling factor
}

def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    
    print("\nData Configuration:")
    for key, value in DATA_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nPreprocessing Configuration:")
    for key, value in PREPROCESS_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nNetwork Architecture:")
    for key, value in NETWORK_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nOutput Configuration:")
    for key, value in OUTPUT_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("=" * 60)

if __name__ == "__main__":
    print_config()
