# Loan Approval Prediction

A machine learning project that predicts whether a loan application  
should be approved based on historical applicant data.  

## Getting Started

Install dependencies:

```bash
uv sync
```

Run the Logistic Regression model:

```bash
uv run python src/logReg.py
```

Run the XGBoost model:

```bash
uv run python src/xgBoost.py
```

## What This Project Does

The project takes raw applicant data and builds two machine learning models  
to predict loan approval decisions.

**Data preprocessing** (`src/preprocessing.py`) cleans the dataset by  
removing invalid rows, handling missing values, and engineering new features.  
It also normalises the data so both models can use the same input.

**Logistic Regression** (`src/logReg.py`) trains a linear classification  
model and generates evaluation plots.

**XGBoost** (`src/xgBoost.py`) trains an ensemble model and generates  
evaluation plots including feature importance.

## Outputs

After running either model, you will find:

- Trained model files (`.pkl`) in `outputs/`
- Confusion matrix plots
- ROC curve plots
- Metrics tables
- Feature importance plot (XGBoost only)

All outputs are saved to the `outputs/` directory.
