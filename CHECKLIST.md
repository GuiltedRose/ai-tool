# BCI Training Checklist

## Environment

- [ ] Python 3.11.9 available on the training server.
- [ ] Dependencies installed with `pip install -r requirements.txt`.
- [ ] `numpy`, `scipy`, and `biosppy` import successfully.

## Data

- [ ] BNCI dataset config exists for the session index being trained.
- [ ] Remote BNCI `.mat` file downloads successfully, or local `eeg_data.mat` / `eeg_data.npz` exists.
- [ ] Raw `X` is signal or feature data, not metadata.
- [ ] Raw `y` is class labels, not class-name metadata.
- [ ] `X.shape[0] == y.shape[0]` after loading.
- [ ] At least two classes are present.

## Shapes

- [ ] Epoch data starts as `(samples, channels, timepoints)`.
- [ ] Feature data starts as `(samples, features)`.
- [ ] Network input prints as `(samples, depth, height, width)`.
- [ ] Encoded labels print as `(samples, classes, 1)`.
- [ ] Flattened input size is reasonable for the server.

## Signal Processing

- [ ] Bandpass filtering only runs on arrays with enough timepoints.
- [ ] Short 2D feature rows skip time-series filtering.
- [ ] Channel-aware normalization runs before model shaping.

## Training

- [ ] Class weights print before epoch logs.
- [ ] Loss uses weighted categorical cross-entropy.
- [ ] Softmax outputs class probabilities.
- [ ] Saved weights are skipped when shapes do not match the current dataset.

## Common Failures

### Dataset index not implemented

Add the missing BNCI dataset config or reduce `num_runs` to the number of configured datasets.

### Wrong BNCI fields selected

Check the printed `Selected X field` and `Selected y field`. They should usually be `X` and `y`.

### Network shape errors

Check the raw shape and network input shape. The model expects one sample at a time shaped `(depth, height, width)`.

### One class found

The selected label field is probably metadata, constant labels, or an unlabeled stream.
