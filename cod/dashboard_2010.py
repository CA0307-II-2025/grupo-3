import pandas as pd
import numpy as np

df = pd.read_csv(r"data/clean/dashboard_2010_clean.csv")
print(df.head())

print(df.shape)
print(df.dtypes)

missing = df.isna().sum().reset_index()
missing.columns = ["Variable", "NAs"]
missing["%"] = (missing["NAs"] / len(df) * 100).round(2)
missing.to_csv("data//missing_summary.csv", index=False)

outlier_report = []

for col in df.select_dtypes(include=[np.number]):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report.append([col, n_outliers, round(n_outliers / len(df) * 100, 2)])

outlier_df = pd.DataFrame(outlier_report, columns=["Variable", "Outliers", "%"])
outlier_df.to_csv("data//outliers_summary.csv", index=False)


df_missing = pd.read_csv(r"data/missing_summary.csv")
print(df)
