# Predictive Maintenance - NASA C-MAPSS Aircraft Engine RUL Prediction

This project implements Remaining Useful Life (RUL) prediction for NASA's C-MAPSS aircraft engine datasets (FD001, FD002, FD003, FD004) using Random Forest regression with advanced feature engineering.

## Dataset Overview

| Dataset | Train Engines | Test Engines | Conditions | Fault Modes |
|---------|--------------|--------------|------------|-------------|
| FD001 | 100 | 100 | 1 (Sea Level) | 1 (HPC Degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC Degradation) |
| FD003 | 100 | 100 | 1 (Sea Level) | 2 (HPC + Fan Degradation) |
| FD004 | 248 | 249 | 6 | 2 (HPC + Fan Degradation) |

Each engine run ends at failure. Training data has full RUL labels; test data ends before failure with true RUL provided separately.

## Project Structure

```
├── config.py                    # Dataset configurations (trend sensors, RF params, RUL clip)
├── data_preprocessing.py        # Data loading, validation, RUL computation
├── feature_engineering.py       # Rolling stats, slopes, EWMA, condition-aware normalization
├── shared_features.py           # Shared feature transforms (EWMA, diff2, drop rolling)
├── shared_eval.py               # Per-bin evaluation metrics
├── shared_train.py              # Shared training/validation/retrain logic
├── baseline.py                  # Baseline models (Linear Regression for FD001, RF for FD002)
├── hyperparameter.py            # Randomized hyperparameter search with weighted RMSE
├── final_model_fd001.py         # FD001 validation with train/val split
├── final_model_fd002.py         # FD002 validation with train/val split
├── final_retrain_fd001.py       # FD001 retrain on full train, evaluate on test
├── final_retrain_fd002.py       # FD002 retrain on full train, evaluate on test
├── model_selection_fd001.py     # Model comparison (Ridge, Lasso, RF, GB)
├── eda_fd001.py / eda_fd002.py  # Exploratory data analysis
├── error_analysis.py            # Prediction error analysis
├── utils/                       # Shared utilities
│   ├── preprocessing.py         # Data loading, validation, clipping
│   ├── features.py              # Rolling, trend, EWMA feature computation
│   ├── normalization.py         # Global and per-condition scaling
│   ├── eda.py                   # EDA helpers
│   └── validation.py            # Stratified engine split
└── test_*.py                    # Unit tests
```

## Key Features

### Feature Engineering (State 2: EWMA replaces Rolling)
- **Rolling statistics**: Mean/std over windows [5, 20] - *dropped in State 2*
- **Slope features**: Linear trend over windows [5, 20] - *retained*
- **EWMA features**: Exponential weighted moving average over spans [5, 20] - *added*
- **Second-diff EWMA**: Acceleration features (diff of diff of EWMA) - *added*
- **Condition-aware normalization**: KMeans clustering on operational settings (FD002/FD004)
- **Per-condition constant sensor removal**

### Model Configuration
| Dataset | n_estimators | max_depth | min_samples_split | min_samples_leaf | max_features |
|---------|-------------|-----------|-------------------|------------------|--------------|
| FD001 | 406 | 24 | 6 | 3 | sqrt |
| FD002 | 221 | 27 | 26 | 10 | None |

Both use `random_state=42`, `n_jobs=-1`, RUL clipped at 125.

### Evaluation
- **Primary metrics**: RMSE, MAE on clipped RUL (0-125)
- **Per-bin metrics**: RUL bins of 25 ([0,25), [25,50), [50,75), [75,100), [100,125))
- **Weighted RMSE**: Higher weight on lower RUL bins (maintenance-critical)

## Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn
```

### Run Validation (Train/Val Split)
```bash
# FD002
python final_model_fd002.py

# FD001
python final_model_fd001.py
```

### Run Final Retrain (Full Train → Test)
```bash
# FD002
python final_retrain_fd002.py

# FD001
python final_retrain_fd001.py
```

### Hyperparameter Search
```bash
python hyperparameter.py
```

## Results Summary

### FD002 (Test Set)
| Metric | Value |
|--------|-------|
| **RMSE** | 16.304 |
| **MAE** | 10.588 |

**Per-bin:**
| Bin | RMSE | MAE | Samples |
|-----|------|-----|---------|
| [0, 25) | 10.220 | 6.846 | 787 |
| [25, 50) | 19.488 | 14.428 | 1,799 |
| [50, 75) | 21.835 | 17.869 | 2,631 |
| [75, 100) | 19.617 | 16.445 | 3,614 |
| [100, 125) | 14.953 | 8.827 | 25,160 |

### FD001 (Test Set)
| Metric | Value |
|--------|-------|
| **RMSE** | 14.033 |
| **MAE** | 10.011 |

**Per-bin:**
| Bin | RMSE | MAE | Samples |
|-----|------|-----|---------|
| [0, 25) | 10.028 | 6.781 | 218 |
| [25, 50) | 21.424 | 16.467 | 668 |
| [50, 75) | 21.165 | 17.651 | 973 |
| [75, 100) | 17.239 | 14.237 | 1,335 |
| [100, 125) | 11.966 | 8.326 | 9,902 |

## Adding New Datasets

Add a new `DatasetConfig` to `config.py`:

```python
FD003_CONFIG = DatasetConfig(
    name="FD003",
    trend_sensors=[...],
    rul_clip=125,
    ewma_spans=[5, 20],
    n_conditions=1,
    rf_params={...},
)
DATASET_CONFIGS["FD003"] = FD003_CONFIG
```

Then create entry scripts:
```python
# final_model_fd003.py
from config import get_config
from shared_train import run_validation

if __name__ == "__main__":
    config = get_config("FD003")
    results = run_validation(config)
```

## Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", *Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08)*, Denver CO, Oct 2008.