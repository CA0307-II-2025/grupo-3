# Cálculos de análisis descriptivo

import pandas as pd
import numpy as np

df = pd.read_csv("data/clean/dashboard_2010_clean.csv")


# Información general

print("Dimensiones del dataset:", df.shape)
print("\nPrimeras filas:")
print(df.head())

print("\nInformación de columnas:")
print(df.info())

print("\nResumen estadístico (numéricas):")
print(df.describe().T)

print("\nValores faltantes por columna:")
print(df.isnull().sum())

# Estadísticas de variables categóricas

print("\nDistribución por Estado:")
print(df["State"].value_counts())

print("\nDistribución por Tipo de Institución:")
print(df["School Type"].value_counts())


# Estadísticas descriptivas separadas por tipo de institución

stats_by_type = df.groupby("School Type").agg(
    {
        "FFEL SUBSIDIZED Recipients": ["mean", "median", "max"],
        "FFEL SUBSIDIZED $ of Loans Originated": ["mean", "median", "max"],
        "FFEL UNSUBSIDIZED Recipients": ["mean", "median", "max"],
        "FFEL UNSUBSIDIZED $ of Loans Originated": ["mean", "median", "max"],
        "FFEL PARENT PLUS Recipients": ["mean", "median", "max"],
        "FFEL PARENT PLUS $ of Loans Originated": ["mean", "median", "max"],
        "FFEL GRAD PLUS Recipients": ["mean", "median", "max"],
        "FFEL GRAD PLUS $ of Loans Originated": ["mean", "median", "max"],
    }
)
print("\n=== Estadísticas descriptivas por tipo de institución ===")
print(stats_by_type)


# Participación relativa en el total (market share)

totals = df.groupby("School Type")[
    [
        "FFEL SUBSIDIZED $ of Loans Originated",
        "FFEL UNSUBSIDIZED $ of Loans Originated",
        "FFEL PARENT PLUS $ of Loans Originated",
        "FFEL GRAD PLUS $ of Loans Originated",
    ]
].sum()

shares = totals.div(totals.sum(axis=0), axis=1) * 100
print("\n=== Participación porcentual en el total de cada tipo de préstamo ===")
print(shares)


# Ratios relevantes


# Relación entre monto desembolsado y monto originado
df["Ratio_Subsidized"] = df["FFEL SUBSIDIZED $ of Disbursements"] / df[
    "FFEL SUBSIDIZED $ of Loans Originated"
].replace(0, np.nan)
df["Ratio_Unsubsidized"] = df["FFEL UNSUBSIDIZED $ of Disbursements"] / df[
    "FFEL UNSUBSIDIZED $ of Loans Originated"
].replace(0, np.nan)

ratios_by_type = df.groupby("School Type")[
    ["Ratio_Subsidized", "Ratio_Unsubsidized"]
].mean()
print("\n=== Relación promedio entre desembolso y préstamo originado ===")
print(ratios_by_type)


# Distribución de préstamos extremos (outliers)

q95 = df[
    ["FFEL SUBSIDIZED $ of Loans Originated", "FFEL UNSUBSIDIZED $ of Loans Originated"]
].quantile(0.95)

outliers = df[
    (
        df["FFEL SUBSIDIZED $ of Loans Originated"]
        > q95["FFEL SUBSIDIZED $ of Loans Originated"]
    )
    | (
        df["FFEL UNSUBSIDIZED $ of Loans Originated"]
        > q95["FFEL UNSUBSIDIZED $ of Loans Originated"]
    )
][
    [
        "School",
        "State",
        "School Type",
        "FFEL SUBSIDIZED $ of Loans Originated",
        "FFEL UNSUBSIDIZED $ of Loans Originated",
    ]
]

print("\n=== Instituciones en el 5% superior por préstamos originados ===")
print(outliers.head(10))
