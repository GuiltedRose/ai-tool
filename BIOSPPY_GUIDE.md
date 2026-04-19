# BioSPPy Signal Processing Guide

BioSPPy is used here as part of the signal-processing context for biometric data. The neural network itself is custom NumPy/SciPy code.

## Accepted Data

Use arrays shaped as either:

- `(samples, channels, timepoints)` for EEG epochs.
- `(samples, features)` for feature/channel rows.

Labels should be shaped as:

- `(samples,)`
- `(samples, 1)`

The loader remaps labels to one-hot vectors with shape `(samples, classes, 1)`.

## Preprocessing Behavior

`apply_biosppy_preprocessing()` applies a 1-50 Hz Butterworth bandpass to time-series data.

For 2D arrays with too few columns to be meaningful timepoints, filtering is skipped. This avoids treating feature/channel rows such as `(347704, 8)` as hundreds of thousands of 8-point signals.

## Normalization

`normalize_biometric_data()` uses channel-aware z-scoring:

- 3D epoch data normalizes each channel across samples and time.
- 2D feature rows normalize each feature/channel across samples.
- Other shapes fall back to whole-array z-scoring.

## Model Shape

Before training, data is converted to the custom layer format:

- `(samples, channels, timepoints)` -> `(samples, 1, channels, timepoints)`
- `(samples, features)` -> `(samples, 1, 1, features)`

Each sample is then flattened by the first `Reshape` layer inside the model.

## Dataset-Specific Note

Some BNCI datasets contain continuous streams plus event/trial metadata. For those, the next quality step is dataset-specific epoching around events before training. The current loader selects real `X` and `y` fields and avoids metadata fields such as `trial` and `classes` as model targets.
