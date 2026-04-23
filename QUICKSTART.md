# Quick Start - BCI Neural Network

This project trains a custom NumPy/SciPy neural network on biometric BCI data. The model and training loop are implemented from scratch.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Run BNCI Training

```bash
python main.py
```

The training loop loads configured BNCI Horizon 2020 datasets in deterministic order:

- session 1 loads dataset 1
- session 2 loads dataset 2
- session 32 loads dataset 32 from the official BNCI catalog

For each dataset set, the default run uses one training epoch per sample in that set's training split. That means each of the 32 configured catalog sets is walked once before moving to the next set.

Within each dataset set, every configured `.mat` file in that set's file list is downloaded and merged before training begins.

The current code intentionally does not replace failed BCI loads with unrelated image data.

## 3. Local Data Format

Local files may be named `eeg_data.mat` or `eeg_data.npz`, or passed as a custom path to `load_and_split_data()`.

Expected arrays:

- `x`: `(samples, channels, timepoints)` or `(samples, features)`
- `y`: `(samples,)` or `(samples, 1)`

The preprocessing path converts model inputs to:

- `(samples, 1, channels, timepoints)` for epoch data
- `(samples, 1, 1, features)` for feature rows

Labels are converted to one-hot vectors shaped `(samples, classes, 1)`.

## 4. What To Check First

When a dataset fails, inspect these printed values:

- selected BNCI `X` field
- selected BNCI `y` field
- raw `X shape`
- raw `Y shape`
- network input shape
- encoded label shape
- class weights

For P300-style datasets, class weights are expected to be uneven because target and non-target labels are usually imbalanced.
