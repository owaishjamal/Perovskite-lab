# Retraining Guide: Robust Model for Limited Features

## Overview

The original model was trained on **105 features**, but your form only collects **13 features**. This causes the model to be conservative when most features are at defaults.

This guide shows how to retrain a model specifically on the subset of features available in your form, making it more robust and accurate for your use case.

## Quick Start

1. **Make sure you have the training data:**
   - The script expects: `/Users/rishuraj7581gmail.com/Downloads/Perovskite_database_content_all_data.csv`
   - Update the `DATA_PATH` in `retrain_robust_model.py` if your data is elsewhere

2. **Run the retraining script:**
   ```bash
   cd /Users/rishuraj7581gmail.com/Desktop/mini_project_22222
   python3 retrain_robust_model.py
   ```

3. **The script will:**
   - Train a model on only the 13 features from your form
   - Save the new model to `backend/model_robust_subset.pt`
   - Save the preprocessor to `backend/preprocess_robust_subset.joblib`
   - Save artifacts to `backend/artifacts_robust_subset.json`

4. **Restart your backend:**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
   The backend will automatically detect and use the new robust model!

## What This Does

### Original Model
- Trained on: 105 features (25 numeric + 80 categorical)
- Problem: When you provide only 13 features, 92 are at defaults
- Result: Model is very conservative, predicts low efficiency

### New Robust Model
- Trained on: Only the 13 features available in your form
- Solution: Model learns patterns from these specific features
- Result: More accurate predictions when using the form

## Expected Improvements

With your excellent JV parameters (Jsc=28, Voc=1.6, FF=91.5%):

**Before (original model):**
- Predicted Efficiency: ~5.77%
- Score: ~4.7/100

**After (robust model):**
- Predicted Efficiency: Should be higher (10-15%+)
- Score: Should be 50-80+ depending on stability

## Features Used in Training

The robust model trains on these 13 features:

**Numeric:**
- `Perovskite_band_gap`
- `JV_default_Jsc`
- `JV_default_Voc`
- `JV_default_FF`
- `Perovskite_deposition_thermal_annealing_temperature`
- `Perovskite_deposition_thermal_annealing_time`

**Categorical:**
- `Cell_architecture`
- `Perovskite_composition_short_form`
- `ETL_stack_sequence`
- `HTL_stack_sequence`
- `Perovskite_deposition_procedure`
- `Perovskite_additives_compounds`
- `Encapsulation`

## Model Architecture

The robust model uses:
- Same architecture as original (3-layer MLP with 256 hidden units)
- Added dropout (0.2) for regularization
- Gradient clipping for stability
- Early stopping to prevent overfitting

## Training Details

- **Epochs:** Up to 200 (with early stopping)
- **Batch Size:** 64
- **Learning Rate:** 0.001 (with ReduceLROnPlateau scheduler)
- **Loss Function:** Heteroscedastic loss (predicts mean + uncertainty)
- **Validation Split:** 30% of data
- **Test Split:** 15% of data

## Troubleshooting

**Issue: FileNotFoundError for CSV**
- Update `DATA_PATH` in `retrain_robust_model.py` to point to your data file

**Issue: CUDA out of memory**
- The script automatically uses CPU if CUDA isn't available
- If you have CUDA but run out of memory, reduce batch size in the script

**Issue: Model doesn't improve**
- Check that the features exist in your CSV file
- Some features might have different names - update `form_features` dict in the script

## Switching Back to Original Model

If you want to use the original model again:
```bash
cd backend
mv model_robust_subset.pt model_robust_subset.pt.backup
# Restart backend - it will automatically use model_best.pt
```

## Next Steps

After retraining, test with your form inputs:
- Jsc: 28 mA/cm²
- Voc: 1.6 V
- FF: 91.5%

You should see significantly higher predictions and scores!

