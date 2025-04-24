import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# ----- paths -----
root = Path(__file__).resolve().parents[2]        # 项目根
raw  = root / "data" / "raw" / "IPO_data_2025_S1.csv"
proc = root / "data" / "processed"
proc.mkdir(exist_ok=True)

# ----- read & clean -----
df = (pd.read_csv(raw)
        .dropna()
        .query("exchange in ['Nasdaq', 'New York']"))

df["ipo_date"] = pd.to_datetime(df["ipo_date"], dayfirst=True)
df["ipo_year"] = df["ipo_date"].dt.year
df["ipo_year_2000"] = (df["ipo_year"] <= 2000).astype(int)
df["ipo_year_2009"] = df["ipo_year"].between(2001, 2009).astype(int)
df["ipo_year_2020"] = (df["ipo_year"] >= 2010).astype(int)
df["log_assets"]  = np.log(df["assets"])
df["log_ipo"]     = np.log(df["ipo_amount"])
df["leverage"]    = df["debt"] / df["assets"]
df["roa"]         = df["profit"] / df["assets"]
df["is_ca"]       = (df["state"] == "California").astype(int)
df.to_parquet(proc / "clean_ipo.parquet", index=False)

# ----- regression -----
model = smf.ols(
    'ipo_fees ~ log_assets + leverage + roa + is_ca + C(industry)'
    ' + ipo_year_2009 + ipo_year_2020',
    data=df
).fit(cov_type="HC1")
print(model.summary())

# ----- prediction sample -----
sample = pd.DataFrame(
    {"log_assets":[np.log(500)],
     "leverage":[0.1],
     "roa":[0.05],
     "is_ca":[1],
     "industry":['IT'],
     "ipo_year_2009":[0],
     "ipo_year_2020":[0]}
)
print("Predicted fee %:", model.predict(sample).iloc[0].round(2))
