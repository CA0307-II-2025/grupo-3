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
    dataframe = _safe_log10_series(df["FFEL SUBSIDIZED $ of Loans Originated"])

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
    dataframe = _safe_log10_series(df["FFEL SUBSIDIZED $ of Loans Originated"])

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
    dataframe = _safe_log10_series(df["FFEL UNSUBSIDIZED $ of Loans Originated"])

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


def plot_correlation_heatmap_pearson(df, savepath):
    """
    Heatmap de correlación de pearson entre variables numéricas.
    """
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 8))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlación")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Matriz de correlación entre variables numéricas utilizando Pearson")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_correlation_heatmap_kendall(df, savepath):
    """
    Heatmap de correlación de kendall entre variables numéricas.
    """
    corr = df.corr(method="kendall", numeric_only=True)

    plt.figure(figsize=(12, 8))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlación")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(
        "Matriz de correlación entre variables numéricas utilizando Tau de Kendall"
    )
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_correlation_heatmap_spearman(df, savepath):
    """
    Heatmap de correlación de spearman entre variables numéricas.
    """
    corr = df.corr(method="spearman", numeric_only=True)

    plt.figure(figsize=(12, 8))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlación")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(
        "Matriz de correlación entre variables numéricas utilizando Rho de Spearman"
    )
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_distribution_by_institution_type(
    df, savepath, variable="$originated", use_violin=True
):
    """
    Boxplot o violin plot de la distribución de montos por tipo de institución, en escala log10.

    Parámetros:
        df (pd.DataFrame): DataFrame con columna 'School Type' y variable monetaria.
        savepath (str o Path): Ruta donde guardar la imagen.
        variable (str): Puede ser "$originated" o "$disbursed".
        use_violin (bool): Si True usa violin plot, si False usa boxplot.
    """
    import seaborn as sns

    # Mapear nombre corto a variable de columna
    column_map = {
        "$originated": "FFEL SUBSIDIZED $ of Loans Originated",
        "$disbursed": "FFEL SUBSIDIZED $ of Disbursements",
    }

    if variable not in column_map:
        raise ValueError("variable debe ser '$originated' o '$disbursed'")

    col = column_map[variable]
    df = df.copy()

    # Filtrar valores positivos y aplicar log10
    df["Monto log10"] = _safe_log10_series(df[col])
    df = df[["School Type", "Monto log10"]].dropna()

    plt.figure(figsize=(10, 6))
    if use_violin:
        sns.violinplot(
            data=df, x="School Type", y="Monto log10", inner="box", palette="Set2"
        )
    else:
        sns.boxplot(data=df, x="School Type", y="Monto log10", palette="Set2")

    tipo = "Violin" if use_violin else "Box"
    titulo = f"{tipo} plot del monto {'originado' if variable == '$originated' else 'desembolsado'} (log) por tipo de institución"
    plt.title(titulo)
    plt.xlabel("Tipo de institución")
    plt.ylabel("Log10(Monto en USD)")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Pie de figura descriptivo
    plt.figtext(
        0.5,
        -0.1,
        "Fuente: FFEL 2010. Se muestra la distribución logarítmica de montos por tipo de institución.\n"
        "Medianas y dispersión (whiskers) indicadas en escala logarítmica. Solo préstamos subsidiados.",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_scatter_log_recipients_vs_originated(df, savepath, use_loess=True):
    """
    Gráfico doble:
    - Izquierda: dispersión log–log entre receptores y monto originado, coloreado por tipo.
    - Derecha: curvas de tendencia por tipo (LOESS si se puede, lineal si no).

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas necesarias.
        savepath (str o Path): Ruta donde guardar el gráfico.
        use_loess (bool): Si True, intenta usar suavizado LOESS (requiere statsmodels).
    """
    import seaborn as sns
    from matplotlib import pyplot as plt

    # Preparar datos
    col_x = "FFEL SUBSIDIZED Recipients"
    col_y = "FFEL SUBSIDIZED $ of Loans Originated"
    col_hue = "School Type"

    df = df[[col_x, col_y, col_hue]].dropna()
    df = df[(df[col_x] > 0) & (df[col_y] > 0)].copy()
    df["log_x"] = np.log10(df[col_x])
    df["log_y"] = np.log10(df[col_y])

    tipos = df[col_hue].unique()
    palette = sns.color_palette("Set2", n_colors=len(tipos))
    color_dict = dict(zip(tipos, palette))

    # Crear figura doble
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # -------- Panel 1: Dispersión
    for tipo in tipos:
        subset = df[df[col_hue] == tipo]
        axs[0].scatter(
            subset["log_x"],
            subset["log_y"],
            label=tipo,
            alpha=0.6,
            color=color_dict[tipo],
        )
    axs[0].set_title("Dispersión log–log por tipo de institución")
    axs[0].set_xlabel("Log10(Número de receptores)")
    axs[0].set_ylabel("Log10(Monto originado en USD)")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend()

    # -------- Panel 2: Tendencias
    for tipo in tipos:
        subset = df[df[col_hue] == tipo]
        if len(subset) >= 5:
            sns.regplot(
                data=subset,
                x="log_x",
                y="log_y",
                scatter=False,
                ax=axs[1],
                label=tipo,
                color=color_dict[tipo],
                lowess=use_loess,  # esto requiere statsmodels si es True
                line_kws={"linestyle": "--", "linewidth": 2},
            )
    axs[1].set_title("Curvas de tendencia por tipo")
    axs[1].set_xlabel("Log10(Número de receptores)")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend()

    # Pie de figura explicativo
    fig.suptitle(
        "Relación entre número de receptores y monto originado (FFEL Subsidized, log–log)",
        fontsize=14,
    )
    fig.subplots_adjust(bottom=0.25, top=0.88)

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()
