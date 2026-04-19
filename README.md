# AI Neural Network Implementation for BCI

This project is a from-scratch NumPy/SciPy neural network for biometric BCI classification. It does not use external deep-learning frameworks for model layers, labels, losses, or training.

The data path is BCI-only:

1. Load a configured BNCI Horizon 2020 dataset.
2. Optionally load a local `.mat` or `.npz` EEG/biometric file.
3. Optionally generate synthetic EEG-like data when BioSPPy is installed and BNCI/local data is not required.
4. Preprocess signal data with SciPy/BioSPPy-style filtering.
5. Normalize channel-aware biometric data.
6. Train the custom network with weighted categorical cross-entropy.

## Files

- `main.py` - BNCI/local data loading, signal preprocessing, shaping, training, evaluation, and weight saving.
- `losses.py` - Weighted categorical cross-entropy for one-hot BCI labels.
- `dense.py` - Fully connected layer.
- `activation.py` - Sigmoid and tanh activation layers.
- `reshape.py` - Shape conversion layer.
- `convolution.py` - Optional 2D convolution layer.
- `data_explorer.py` - Dataset inspection and signal preprocessing helpers.
- `config.py` - Project configuration values.
- `requirements.txt` - Runtime dependencies for NumPy/SciPy/BioSPPy workflow.

## Data Shape Contract

Raw biometric inputs may be:

- `(samples, channels, timepoints)` for EEG epochs.
- `(samples, features)` for feature/channel rows.
- `(samples, timepoints)` for single-channel time series.

`main.py` converts these to the custom layer format:

- `(samples, 1, channels, timepoints)` for epoch data.
- `(samples, 1, 1, features)` for feature rows.

Labels are remapped to stable zero-based one-hot vectors shaped `(samples, classes, 1)`.

## Loss And Metrics

The classifier uses:

- Softmax output.
- Weighted categorical cross-entropy.
- Inverse-frequency class weights computed from the training split.
- Accuracy from predicted class index vs. one-hot target index.

The class weighting is important for BCI datasets such as P300 tasks, where non-target labels can dominate.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python main.py
```

The default training loop starts at BNCI dataset 1 and proceeds deterministically by session number. It does not silently substitute unrelated image data if a BCI dataset fails.

By default, each BNCI dataset set runs one training epoch per sample in that set's training split, so the training samples in each 1/32 set are consumed once before the loop advances.

Each BNCI dataset set downloads and merges every configured `.mat` file in that set's file list. Empty MATLAB cells or blocks without usable labels are skipped, while labeled continuous streams are cut into trial epochs when trial markers are available.
