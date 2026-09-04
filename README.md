# MLOps Assignment 1 — Part 1

Baseline ML training pipeline: load a dataset, train a model, and serialize the
trained artifact.

## Project structure

```
├── data/                 # Raw dataset (data.csv — Bank of England base rate history)
├── src/
│   ├── train.py          # Main training script
│   └── train_23L2572.py  # Baseline data-loading module (load_data helper)
├── model/                # Output directory for the trained model artifact
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

`data/data.csv` contains the historical Bank of England base rate with columns
`Date Changed` and `Rate`. The training script engineers date-based features
(year, month, day, day-of-year, previous rate) and trains a regressor to predict
`Rate`.

## Setup

Run all commands from the repository root.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Train the model

```bash
python src/train.py
```

This loads `data/data.csv`, trains a `RandomForestRegressor`, prints evaluation
metrics (MAE and R²), and saves the trained model to `model/model.joblib`.
