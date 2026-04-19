# Implementation Checklist

Use this checklist to verify your BCI neural network is ready to go.

## ✅ Phase 1: Verify Installation

- [ ] Python 3.7+ installed: `python --version`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Can import required packages:
  ```bash
  python -c "import numpy, scipy, tensorflow, keras"
  ```
- [ ] Optional - pyscppy installed: `pip install pyscppy`

## ✅ Phase 2: Verify Code Structure

- [ ] All files present in workspace:
  - [ ] main.py
  - [ ] config.py
  - [ ] data_explorer.py
  - [ ] Dense layers and activations (dense.py, activation.py, etc.)
  - [ ] Requirements.txt with pyscppy listed

- [ ] Documentation files present:
  - [ ] README.md
  - [ ] QUICKSTART.md
  - [ ] CONVERSION_SUMMARY.md
  - [ ] PYSCPPY_INTEGRATION.md (this file)

## ✅ Phase 3: Test with MNIST

- [ ] Run: `python main.py`
- [ ] Verify output shows training progress
- [ ] Check that validation metrics appear each epoch
- [ ] Verify final test accuracy is reported

Expected output pattern:
```
Starting training...
Train samples: 45000, Val samples: 9000, Test samples: 16000
Epoch 10/500
  Train Loss: X.XXXXXX | Train Acc: 0.XXXX
  Val Loss: X.XXXXXX | Val Acc: 0.XXXX
```

## ✅ Phase 4: Explore Your EEG Data

- [ ] Run: `python data_explorer.py`
- [ ] Verify it prints dataset information
- [ ] Check data shapes are compatible
- [ ] Review class distribution

Expected:
```
DATASET INFORMATION
X shape: (n_samples, ...)
Y shape: (n_samples,)
...
```

## ✅ Phase 5: pyscppy Integration

- [ ] Read PYSCPPY_INTEGRATION.md
- [ ] Install pyscppy: `pip install pyscppy`
- [ ] Determine pyscppy's data loading API:
  - [ ] Try: `import pyscppy; help(pyscppy)`
  - [ ] Run test code to explore available functions
  - [ ] Identify correct loading function
- [ ] Update `load_pyscppy_data()` function in main.py
- [ ] Test loading: `python -c "from main import load_pyscppy_data; x, y = load_pyscppy_data()"`

## ✅ Phase 6: Data Shape Verification

- [ ] Print your data shape in Python:
  ```python
  import pyscppy
  x, y = pyscppy.load_dataset()  # your API
  print(f"X shape: {x.shape}, Y shape: {y.shape}")
  ```

- [ ] Update NETWORK_CONFIG in config.py:
  ```python
  'input_shape': (channels, height, width)  # based on your data
  ```

- [ ] Verify network architecture matches your data

## ✅ Phase 7: First Training Run

- [ ] Run: `python main.py`
- [ ] Verify training starts without errors
- [ ] Check first epoch completes
- [ ] Verify metrics include Train/Val/Test
- [ ] Let it run for at least 2-3 epochs to verify stability

## ✅ Phase 8: Verify 75/15/10 Split

- [ ] Check training output for sample counts
- [ ] Example output should show:
  ```
  Train samples: X, Val samples: Y, Test samples: Z
  ```
  Where roughly: X ≈ 75%, Y ≈ 15%, Z ≈ 10% of total

- [ ] Verify each set is separate by checking accuracy patterns
- [ ] Validation accuracy should be tracked separately from training

## ✅ Phase 9: Configure for Your Needs

- [ ] Edit config.py to adjust:
  - [ ] Number of epochs
  - [ ] Learning rate
  - [ ] Network architecture
  - [ ] Data preprocessing options

- [ ] Re-run training: `python main.py`
- [ ] Monitor if changes improve validation accuracy

## ✅ Phase 10: Production Ready

- [ ] Training runs successfully on your EEG data
- [ ] Validation metrics are being tracked
- [ ] Model weights are being saved to weights.npz
- [ ] Training curves show expected patterns (loss decreasing, accuracy increasing)
- [ ] Final test accuracy is reported

## 🐛 Troubleshooting

If you encounter issues, check:

### Won't start
- [ ] Dependencies installed? `pip install -r requirements.txt`
- [ ] Python version correct? `python --version`
- [ ] Can import main modules? `python -c "import main"`

### MNIST test fails
- [ ] Check error message for missing imports
- [ ] Verify TensorFlow/Keras installed: `pip install tensorflow keras`

### pyscppy data fails to load
- [ ] Is pyscppy installed? `pip list | grep pyscppy`
- [ ] Are you in the right directory? `pwd`
- [ ] Check `load_pyscppy_data()` function implementation
- [ ] Run: `python data_explorer.py` to debug

### Network shape errors
- [ ] Check your actual data shape
- [ ] Update input_shape in config.py
- [ ] Print shapes before running: `print(x_train.shape)`

### Poor accuracy / Training too slow
- [ ] Try adjusting learning_rate in config.py
- [ ] Increase epochs in config.py
- [ ] Check if data is properly normalized in preprocessing
- [ ] Verify labels are correctly formatted (0-indexed, no gaps)

## 📊 Expected Performance

- **MNIST Test**: 
  - Initial loss: ~2.3 (random)
  - After 10 epochs: ~0.1-0.5
  - After 100 epochs: Should see 90%+ accuracy

- **EEG Data**: 
  - Will vary based on dataset quality and class separability
  - Monitor validation accuracy to detect overfitting
  - Should see loss decreasing over time

## ✨ You're Ready When:

✅ All checkboxes in Phases 1-10 are complete
✅ Training runs without errors
✅ Metrics are tracked (train/val/test)
✅ Data split is 75/15/10
✅ You understand how to adjust hyperparameters

## 🎉 Success Indicators

- [ ] Training starts: "Starting training..."
- [ ] Data loaded: "Train samples: X, Val samples: Y, Test samples: Z"
- [ ] Epochs progress: "Epoch 10/500"
- [ ] Metrics appear: "Train Loss: X | Val Loss: Y"
- [ ] Final report: "Final Evaluation on Test Set"
- [ ] Weights saved: "weights.npz" updated

---

**Start here:** Run `python main.py` to verify basic functionality
**Next step:** Update pyscppy integration (see PYSCPPY_INTEGRATION.md)
**Final step:** Monitor training metrics and adjust as needed
