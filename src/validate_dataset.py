from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[1] / "data" / "umkm_success.csv"
df = pd.read_csv(path)
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("missing values:")
print(df.isna().sum())
print("target distribution:")
print(df["Success"].value_counts())
print("correlation with Success:")
print(df.corr(numeric_only=True)["Success"].sort_values(ascending=False))
