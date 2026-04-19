# BCI Cleanup Summary

The project is now aligned around a single direction: custom NumPy/SciPy neural-network training for BNCI/local biometric BCI data.

## Current State

- Model layers are implemented from scratch.
- Label encoding is implemented with NumPy.
- Loss is weighted categorical cross-entropy for one-hot BCI labels.
- Softmax backward is matched to the softmax plus cross-entropy gradient.
- Data loading is BCI-only.
- BNCI sessions start deterministically from dataset 1.
- Signal preprocessing skips invalid short 2D feature rows.
- Normalization preserves channel-specific statistics.

## Data Flow

```text
BNCI or local biometric file
  -> selected signal field X and label field y
  -> optional signal filtering
  -> channel-aware normalization
  -> network shape conversion
  -> one-hot label encoding
  -> class-weighted training
```

## Shape Contract

- Epoch signals: `(samples, channels, timepoints)`
- Feature rows: `(samples, features)`
- Network samples: `(depth, height, width)`
- Batched network input: `(samples, depth, height, width)`
- Encoded labels: `(samples, classes, 1)`

## Remaining Work

- Add the remaining BNCI dataset configs if the loop should truly run all 32 sessions.
- Decide whether each BNCI dataset should train on continuous rows, epochs around events, or extracted features.
- Add dataset-specific epoching for event-based tasks such as P300 if raw streams are not already segmented.
