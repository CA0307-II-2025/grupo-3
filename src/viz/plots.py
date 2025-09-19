# Generación de gráficos (histogramas, dispersión, etc.)
# src/viz/plots.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_dashboard_csv(filepath):
    """
    Carga el CSV con los datos del dashboard 2010.

    Parámetros:
        filepath (str o Path): Ruta al archivo CSV.

    Retorna:
        pd.DataFrame: Datos cargados.
    """
    return pd.read_csv(filepath)


def split_public_private(df):
    """
    Separa el DataFrame en dos: instituciones públicas y privadas.

    Parámetros:
        df (pd.DataFrame): DataFrame original.

    Retorna:
        tuple: (DataFrame públicas, DataFrame privadas)
    """
    publicas = df[df["School Type"].str.upper() == "PUBLIC"]
    privadas = df[df["School Type"].str.upper() == "PRIVATE"]
    return publicas, privadas


def plot_hist_subsidized_public(df, savepath):
    """
    Histograma de préstamos subsidiados originados (monto) en universidades públicas.

    Parámetros:
        df (pd.DataFrame): Datos de instituciones públicas.
        savepath (str o Path): Ruta donde guardar la imagen PNG.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(
        df["FFEL SUBSIDIZED $ of Loans Originated"],
        bins=40,
        color="skyblue",
        edgecolor="black",
    )
    plt.title("Préstamos subsidiados originados - Instituciones públicas")
    plt.xlabel("Monto de préstamos originados (USD)")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath)


def plot_hist_subsidized_private(df, savepath):
    """
    Histograma de préstamos subsidiados originados (monto) en universidades privadas.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(
        df["FFEL SUBSIDIZED $ of Loans Originated"],
        bins=40,
        color="orange",
        edgecolor="black",
    )
    plt.title("Préstamos subsidiados originados - Instituciones privadas")
    plt.xlabel("Monto de préstamos originados (USD)")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath)


def plot_hist_unsubsidized_public(df, savepath):
    """
    Histograma de préstamos NO subsidiados originados (monto) en universidades públicas.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(
        df["FFEL UNSUBSIDIZED $ of Loans Originated"],
        bins=40,
        color="green",
        edgecolor="black",
    )
    plt.title("Préstamos no subsidiados originados - Instituciones públicas")
    plt.xlabel("Monto de préstamos originados (USD)")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath)


def plot_hist_unsubsidized_private(df, savepath):
    """
    Gráfico de barras comparando el monto total de préstamos subsidiados y no subsidiados.
    """
    totals = {
        "Subsidized": df["FFEL SUBSIDIZED $ of Loans Originated"].sum(),
        "Unsubsidized": df["FFEL UNSUBSIDIZED $ of Loans Originated"].sum(),
    }

    plt.figure(figsize=(8, 6))
    plt.bar(totals.keys(), totals.values(), color=["skyblue", "salmon"])
    plt.title("Total de préstamos originados: subsidiados vs no subsidiados")
    plt.ylabel("Monto total (USD)")
    plt.tight_layout()
    plt.savefig(savepath)
    

def _safe_log10_series(s: pd.Series) -> pd.Series:
    """
    Aplica log10 a valores positivos (x > 0).

    Retorna una Serie en escala log10 con índice preservado.
    """
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    if s.empty:
        return s.astype(float)
    x = np.log10(s)
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    return x
    
    
def plot_log_hist_subsidized_public(df, savepath):
    """
    Histograma aplicando log10 a los préstamos subsidiados originados (monto) en universidades públicas.

    Parámetros:
        df (pd.DataFrame): Datos de instituciones públicas.
        savepath (str o Path): Ruta donde guardar la imagen PNG.
    """
    dataframe=_safe_log10_series(df["FFEL SUBSIDIZED $ of Loans Originated"])
    
    plt.figure(figsize=(10, 6))
    plt.hist(
        dataframe,
        bins=100,
        color="skyblue",
        edgecolor="black",
    )
    plt.title("Préstamos subsidiados originados (Log) - Instituciones públicas")
    plt.xlabel("Log(Monto de préstamos originados (USD))")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath)


def plot_log_hist_subsidized_private(df, savepath):
    """
    Histograma aplicando log10 a los préstamos subsidiados originados (monto) en universidades privadas.
    """
    dataframe=_safe_log10_series(df["FFEL SUBSIDIZED $ of Loans Originated"])
    
    plt.figure(figsize=(10, 6))
    plt.hist(
        dataframe,
        bins=100,
        color="orange",
        edgecolor="black",
    )
    plt.title("Préstamos subsidiados originados (Log) - Instituciones privadas")
    plt.xlabel("Log(Monto de préstamos originados (USD))")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath)


def plot_log_hist_unsubsidized_public(df, savepath):
    """
    Histograma aplicando log10 a los préstamos NO subsidiados originados (monto) en universidades públicas.
    """
    dataframe=_safe_log10_series(df["FFEL UNSUBSIDIZED $ of Loans Originated"])
    
    plt.figure(figsize=(10, 6))
    plt.hist(
        dataframe,
        bins=100,
        color="green",
        edgecolor="black",
    )
    plt.title("Préstamos no subsidiados originados (Log) - Instituciones públicas")
    plt.xlabel("Log(Monto de préstamos originados (USD))")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath)


def plot_total_subsidized_vs_unsubsidized(df, savepath):
    """
    Gráfico de barras comparando el monto total de préstamos subsidiados y no subsidiados.
    """
    totals = {
        "Subsidized": df["FFEL SUBSIDIZED $ of Loans Originated"].sum(),
        "Unsubsidized": df["FFEL UNSUBSIDIZED $ of Loans Originated"].sum(),
    }

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.bar(totals.keys(), totals.values(), color=["skyblue", "salmon"])
    plt.title("Total de préstamos originados: subsidiados vs no subsidiados")
    plt.ylabel("Monto total (USD)")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_loan_distribution_by_school_type(df, savepath):
    """
    Pie chart de la distribución total de préstamos (sub + unsub) por tipo de institución.
    """
    df["Total Loans"] = (
        df["FFEL SUBSIDIZED $ of Loans Originated"]
        + df["FFEL UNSUBSIDIZED $ of Loans Originated"]
    )
    grouped = df.groupby("School Type")["Total Loans"].sum()

    plt.figure(figsize=(8, 8))
    plt.pie(grouped, labels=grouped.index, autopct="%1.1f%%", startangle=140)
    plt.title("Distribución de préstamos originados por tipo de institución")
    plt.tight_layout()
    plt.savefig(savepath)


def plot_scatter_recipients_vs_amount(df, savepath):
    """
    Diagrama de dispersión entre número de receptores y monto de préstamos subsidiados.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["FFEL SUBSIDIZED Recipients"],
        df["FFEL SUBSIDIZED $ of Loans Originated"],
        alpha=0.6,
    )
    plt.title("Receptores vs monto de préstamos subsidiados")
    plt.xlabel("Número de receptores")
    plt.ylabel("Monto total originado (USD)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath)


def plot_top_states_by_subsidized_loans(df, savepath, top_n=10):
    """
    Barras horizontales con los 10 estados con mayores montos de préstamos subsidiados.
    """
    grouped = df.groupby("State")["FFEL SUBSIDIZED $ of Loans Originated"].sum()
    top_states = grouped.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    top_states.sort_values().plot(kind="barh", color="mediumseagreen")
    plt.title(f"Top {top_n} estados por monto de préstamos subsidiados")
    plt.xlabel("Monto total (USD)")
    plt.tight_layout()
    plt.savefig(savepath)


def plot_institutions_by_state(df, savepath):
    """
    Gráfico de barras con el número de instituciones por estado.
    """
    counts = df["State"].value_counts()

    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.xticks(rotation=90)
    plt.title("Número de instituciones por Estado")
    plt.xlabel("Estado")
    plt.ylabel("Número de instituciones")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_institutions_by_type(df, savepath):
    """
    Gráfico de barras con la distribución de instituciones por tipo (Public/Private/etc).
    """
    counts = df["School Type"].value_counts()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color="darkorange", edgecolor="black")
    plt.title("Distribución por Tipo de Institución")
    plt.xlabel("Tipo de Institución")
    plt.ylabel("Número de instituciones")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_scatter_subsidized_vs_disbursements(df, savepath):
    """
    Diagrama de dispersión entre préstamos subsidiados y desembolsos totales.
    Se colorea según el tipo de institución.
    """
    if "US $ of Disbursements" in df.columns and "FFEL SUBSIDIZED" in df.columns:
        plt.figure(figsize=(8, 6))
        types = df["School Type"].unique()

        for t in types:
            subset = df[df["School Type"] == t]
            plt.scatter(
                subset["FFEL SUBSIDIZED"],
                subset["US $ of Disbursements"],
                alpha=0.6,
                label=t,
            )

        plt.title("Relación entre préstamos subsidiados y desembolsos")
        plt.xlabel("FFEL SUBSIDIZED (USD)")
        plt.ylabel("US $ of Disbursements (USD)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(savepath)
        plt.close()


def plot_correlation_heatmap(df, savepath):
    """
    Heatmap de correlación entre variables numéricas.
    """
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 8))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlación")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Matriz de correlación entre variables numéricas")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
