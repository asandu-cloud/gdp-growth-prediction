# Entropy of the Yield Curve as a Macroeconomic Growth Signal

Data Pipeline. Tests whether Shannon and Tsallis entropy of the US Treasury yield curve predicts real GDP growth.

## Pipeline

| Layer | Script | Description |
|-------|--------|-------------|
| 1 | `layer1_data_acquisition.py` | Download yield curve, GDP, and benchmark series from FRED |
| 2 | `layer2_preprocessing.py` | Clean missing data, align to quarterly frequency |
| 3 | `layer3_entropy.py` | Compute Shannon and Tsallis entropy (Methods A & B) |
| 4 | `layer4_derived_features.py` | Build momentum, z-score, volatility, and other derived features |
| 5 | `layer5_models.py` | OLS, quantile regression, random forest, MIDAS, probit |

## Setup

```bash
cd thesis_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your FRED API key in `config.py` or as an environment variable:

```bash
export FRED_API_KEY="your_key_here"
```

## Sample

1982 Q1 -- 2025 Q1 (~172 quarterly observations). Eight constant-maturity Treasury maturities: 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y.