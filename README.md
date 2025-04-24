# IPO Underwriting-Fee Analytics & Prediction

**Tech stack:** Python · pandas · statsmodels · SAS Visual Analytics  

## Overview  
Cleaned 728 US IPO records and built three OLS models (Adj. R² = 0.60) to predict underwriting fee %. Delivered a 10-page business report and dashboards.

## Key Result  
IT firm (assets $500 m, proceeds $200 m) → **6.8 %** predicted fee.
## Reproduce

```bash
pip install -r requirements.txt
python src/clean_regression.py
