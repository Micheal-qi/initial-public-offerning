# IPO Underwriting-Fee Analytics & Prediction

**Problem solved**  
Underwriting fees vary widely; the project builds a data-driven model that explains **60 % of fee variance** and predicts a fair % for new IPOs.

**Solution**  
1. Clean raw SEC IPO dataset → feature-engineer ROA, leverage, size dummies  
2. Visualise patterns in SAS Visual Analytics  
3. Fit 3 OLS models (Python/statsmodels) with industry & time fixed-effects  
4. Deliver fee estimate & sensitivity analysis in a business report

**Tech stack** Python · pandas · statsmodels · SAS Visual Analytics

## Repo structure
data/               raw & processed datasets
src/clean_regression.py      ⟵ reproducible ETL + modelling script
notebooks/           EDA & chart exploration
report/IPO_fee_report.docx   full write-up
requirements.txt     minimal Python deps

## Quick start
```bash
git clone https://github.com/Micheal-qi/initial-public-offerning.git
cd initial-public-offerning
pip install -r requirements.txt
python src/clean_regression.py

Example prediction → IT firm, assets $500 m, proceeds $200 m = 6.8 % fee.
