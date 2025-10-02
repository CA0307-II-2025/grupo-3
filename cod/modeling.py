# -*- coding: utf-8 -*-
"""
modeling.py — Validación e Inferencia (👤 Persona 3)

Asume que los datos están en:
    C:\\Users\\gsana\\Documents\\grupo-3\\data\\clean

Requiere:
    pandas, numpy, matplotlib, statsmodels, scipy
    (opcional) scikit-learn para train/test split

Funciones principales:
    - load_dataset_from_dir
    - fit_model (OLS o Logit con statsmodels)
    - compute_metrics (LogLik, AIC, BIC)
    - plot_diagnostics (residuos vs pred, QQ)
    - infer_params (coef, SE, IC95%, p-val)
    - dependence_by_group (correlaciones por grupo)
    - compare_dependence_diff (diferencia + IC por bootstrap)
    - evaluate_models_across_groups (pipeline completo)

Autor: Persona 3 — Validación e Inferencia
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# 0) Carga de datos
# =========================


def load_dataset_from_dir(
    dir_path: str = r"C:\Users\gsana\Documents\grupo-3\data\clean",
    pattern: str = "*.csv",
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Lee todos los CSV de un directorio y los concatena.
    Devuelve un DataFrame único.
    """
    path = Path(dir_path)
    files = sorted(glob.glob(str(path / pattern)))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en: {dir_path}")

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        if index_col and index_col in df.columns:
            df.set_index(index_col, inplace=True)
        frames.append(df)
    data = pd.concat(frames, axis=0, ignore_index=True)
    return data


# =========================
# 1) Ajuste de modelos
# =========================


def fit_model(
    df: pd.DataFrame, formula: str, family: str = "ols", add_const: bool = False
):
    """
    Ajusta un modelo con statsmodels usando fórmula:
        - family="ols"  -> smf.ols
        - family="logit"-> smf.logit
    """
    if family.lower() == "ols":
        model = smf.ols(formula=formula, data=df)
        results = model.fit()
    elif family.lower() == "logit":
        model = smf.logit(formula=formula, data=df)
        results = model.fit(disp=0)
    else:
        raise ValueError("family debe ser 'ols' o 'logit'.")

    if add_const:
        exog = results.model.exog
        if not np.any(np.all(exog == 1.0, axis=0)):
            y = results.model.endog
            X = sm.add_constant(results.model.exog)
            if family.lower() == "ols":
                results = sm.OLS(y, X).fit()
            else:
                results = sm.Logit(y, X).fit(disp=0)
    return results


# =========================
# 2) Métricas de ajuste
# =========================


def compute_metrics(results) -> Dict[str, float]:
    """
    Calcula LogLik, AIC, BIC desde el objeto de resultados de statsmodels.
    """
    out = {}
    out["LogLik"] = float(results.llf) if hasattr(results, "llf") else np.nan
    out["AIC"] = float(getattr(results, "aic", np.nan))
    out["BIC"] = float(getattr(results, "bic", np.nan))
    try:
        out["nobs"] = int(results.nobs)
    except Exception:
        out["nobs"] = np.nan
    try:
        out["k_params"] = int(results.df_model) + 1  # + intercept
    except Exception:
        out["k_params"] = getattr(results, "params", np.array([])).shape[0]
    return out


# =========================
# 3) Diagnóstico gráfico
# =========================


def _get_predictions_and_residuals(
    results, df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve y_hat y residuos (y - y_hat) compatible con OLS y Logit.
    Para Logit, usa la probabilidad predicha (para gráficos).
    """
    if hasattr(results, "predict"):
        y_hat = results.predict(df)
    else:
        y_hat = results.fittedvalues

    try:
        y_true = results.model.endog
    except Exception:
        raise ValueError("No se pudo recuperar la variable respuesta del modelo.")

    resid = y_true - y_hat
    return np.asarray(y_hat), np.asarray(resid)


def plot_diagnostics(results, df: pd.DataFrame, title_prefix: str = "") -> None:
    """
    Genera: Residuos vs Predicción y Q-Q plot de residuos (matplotlib puro).
    """
    y_hat, resid = _get_predictions_and_residuals(results, df)

    plt.figure(figsize=(6, 4))
    plt.scatter(y_hat, resid, alpha=0.6)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicción")
    plt.ylabel("Residuo")
    plt.title(f"{title_prefix}Residuos vs Predicción")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title(f"{title_prefix}Q–Q Plot de Residuos")
    plt.tight_layout()
    plt.show()


# =========================
# 4) Inferencia sobre parámetros
# =========================


def infer_params(results) -> pd.DataFrame:
    """
    Devuelve DataFrame con coef, std err, IC95% y p-val.
    Compatible con OLS/Logit de statsmodels.
    """
    params = results.params
    se = results.bse
    ci = results.conf_int(alpha=0.05)
    pvals = results.pvalues

    out = pd.DataFrame(
        {
            "coef": params,
            "std_err": se,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "p_value": pvals,
        }
    )
    out.index.name = "parameter"
    return out


# =========================
# 5) Dependencia público vs privado
# =========================


def dependence_by_group(
    df: pd.DataFrame,
    cols: List[str],
    group_col: str,
    groups: Tuple[str, str] = ("PUBLIC", "PRIVATE"),
    method: str = "pearson",
) -> Dict[str, pd.DataFrame]:
    """
    Calcula matrices de correlación por grupo para columnas numéricas.
    """
    out = {}
    for g in groups:
        sub = df.loc[df[group_col].astype(str) == g, cols].dropna()
        if sub.empty:
            out[g] = pd.DataFrame(np.nan, index=cols, columns=cols)
        else:
            out[g] = sub.corr(method=method)
    return out


def _corr_pair(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "pearson":
        return stats.pearsonr(x, y)[0]
    elif method == "spearman":
        return stats.spearmanr(x, y)[0]
    elif method == "kendall":
        return stats.kendalltau(x, y)[0]
    else:
        raise ValueError("method debe ser 'pearson', 'spearman' o 'kendall'.")


def compare_dependence_diff(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    groups: Tuple[str, str] = ("PUBLIC", "PRIVATE"),
    method: str = "pearson",
    n_boot: int = 2000,
    random_state: Optional[int] = 123,
) -> Dict[str, float]:
    """
    Compara la dependencia (correlación) entre dos grupos y estima un IC95% por bootstrap.
    """
    rng = np.random.default_rng(random_state)

    g1, g2 = groups
    d1 = df.loc[df[group_col].astype(str) == g1, [x_col, y_col]].dropna()
    d2 = df.loc[df[group_col].astype(str) == g2, [x_col, y_col]].dropna()

    if len(d1) < 5 or len(d2) < 5:
        raise ValueError("Muy pocos datos por grupo para comparar dependencia.")

    r1 = _corr_pair(d1[x_col].values, d1[y_col].values, method)
    r2 = _corr_pair(d2[x_col].values, d2[y_col].values, method)
    diff = r2 - r1

    boots = []
    for _ in range(n_boot):
        s1 = d1.sample(len(d1), replace=True, random_state=int(rng.integers(1e9)))
        s2 = d2.sample(len(d2), replace=True, random_state=int(rng.integers(1e9)))
        b1 = _corr_pair(s1[x_col].values, s1[y_col].values, method)
        b2 = _corr_pair(s2[x_col].values, s2[y_col].values, method)
        boots.append(b2 - b1)

    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    return {
        f"r_{g1}": r1,
        f"r_{g2}": r2,
        "diff_r2_minus_r1": diff,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "method": method,
    }


# =========================
# 6) Pipeline de evaluación
# =========================


def evaluate_models_across_groups(
    df: pd.DataFrame,
    formulas: Dict[str, str],
    family: str,
    group_col: str,
    groups: Tuple[str, str] = ("PUBLIC", "PRIVATE"),
    diag_plots: bool = True,
) -> pd.DataFrame:
    """
    Ajusta y evalúa modelos por grupo y devuelve una tabla con LogLik, AIC, BIC.
    """
    rows = []
    for name, formula in formulas.items():
        for g in groups:
            sub = df[df[group_col].astype(str) == g].copy()
            sub = sub.dropna()
            if len(sub) < 10:
                print(f"[ADVERTENCIA] Muy pocos datos para {g} en {name}. Saltando.")
                continue

            res = fit_model(sub, formula=formula, family=family)
            metrics = compute_metrics(res)
            metrics["grupo"] = g
            metrics["modelo"] = name
            rows.append(metrics)

            if diag_plots:
                plot_diagnostics(res, sub, title_prefix=f"[{name} | {g}] ")

            infer = infer_params(res)
            print(f"\n=== Inferencia: {name} | {g} ===")
            print(infer, "\n")

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    cols_metricas = ["LogLik", "AIC", "BIC", "nobs", "k_params"]
    out_pivot = out.pivot_table(index="modelo", columns="grupo", values=cols_metricas)
    out_pivot = out_pivot.sort_index(axis=0)
    return out_pivot


# =========================
# 7) Ejemplo de uso
# =========================

if __name__ == "__main__":
    # 1) Cargar datos
    df = load_dataset_from_dir(
        dir_path=r"C:\Users\gsana\Documents\grupo-3\data\clean", pattern="*.csv"
    )

    # ---- Columna de grupo y grupos a comparar (dataset 2010) ----
    GROUP_COL = "School Type"
    GRUPOS = ("PUBLIC", "PRIVATE")  # valores exactos del CSV

    # ---- Fórmulas OLS usando Q("...") para columnas con espacios ----
    # (Puedes comentar las que no existan en tu CSV exacto)
    FORMULAS_OLS = {
        "subs_base": 'Q("FFEL SUBSIDIZED $ of Loans Originated") ~ '
        'Q("FFEL SUBSIDIZED Recipients") + '
        'Q("FFEL SUBSIDIZED # of Loans Originated") + '
        'Q("FFEL SUBSIDIZED # of Disbursements") + '
        "C(State)",
        "unsubs_base": 'Q("FFEL UNSUBSIDIZED $ of Loans Originated") ~ '
        'Q("FFEL UNSUBSIDIZED Recipients") + '
        'Q("FFEL UNSUBSIDIZED # of Loans Originated") + '
        'Q("FFEL UNSUBSIDIZED # of Disbursements") + '
        "C(State)",
        "plus_parent": 'Q("FFEL PARENT PLUS $ of Loans Originated") ~ '
        'Q("FFEL PARENT PLUS Recipients") + '
        'Q("FFEL PARENT PLUS # of Loans Originated") + '
        'Q("FFEL PARENT PLUS # of Disbursements") + '
        "C(State)",
    }

    # # (Opcional) Si tuvieras outcome binario, define FORMULAS_LOGIT y descomenta el bloque de evaluación.
    # FORMULAS_LOGIT = {
    #     "logit_base": 'Q("Some Binary Outcome") ~ Q("Predictor 1") + Q("Predictor 2") + C(State)'
    # }

    # 2) Evaluación OLS
    try:
        tabla_ols = evaluate_models_across_groups(
            df=df,
            formulas=FORMULAS_OLS,
            family="ols",
            group_col=GROUP_COL,
            groups=GRUPOS,
            diag_plots=True,  # pon False para no graficar
        )
        print("\n=== Resumen OLS (LogLik/AIC/BIC por grupo) ===")
        print(tabla_ols)
    except Exception as e:
        print(f"[INFO] OLS no ejecutado: {e}")

    # # 2b) (Opcional) Evaluación Logit
    # try:
    #     tabla_logit = evaluate_models_across_groups(
    #         df=df,
    #         formulas=FORMULAS_LOGIT,
    #         family="logit",
    #         group_col=GROUP_COL,
    #         groups=GRUPOS,
    #         diag_plots=True
    #     )
    #     print("\n=== Resumen Logit (LogLik/AIC/BIC por grupo) ===")
    #     print(tabla_logit)
    # except Exception as e:
    #     print(f"[INFO] Logit no ejecutado: {e}")

    # 3) Comparación de dependencia (correlaciones) entre grupos
    COLS_DEP = [
        "FFEL SUBSIDIZED Recipients",
        "FFEL SUBSIDIZED # of Loans Originated",
        "FFEL SUBSIDIZED $ of Loans Originated",
        "FFEL SUBSIDIZED # of Disbursements",
        "FFEL SUBSIDIZED $ of Disbursements",
    ]

    try:
        dep = dependence_by_group(
            df=df, cols=COLS_DEP, group_col=GROUP_COL, groups=GRUPOS, method="pearson"
        )
        print("\n=== Correlaciones (Pearson) — PUBLIC ===")
        print(dep["PUBLIC"])
        print("\n=== Correlaciones (Pearson) — PRIVATE ===")
        print(dep["PRIVATE"])
    except Exception as e:
        print(f"[INFO] Dependencia por grupo no ejecutada: {e}")

    # 4) Diferencia en dependencia (ejemplo: $ originado vs receptores) con IC95% (bootstrap)
    try:
        diff_res = compare_dependence_diff(
            df=df,
            x_col="FFEL SUBSIDIZED Recipients",
            y_col="FFEL SUBSIDIZED $ of Loans Originated",
            group_col=GROUP_COL,
            groups=GRUPOS,
            method="pearson",
            n_boot=2000,
            random_state=123,
        )
        print("\n=== Diferencia de correlación (PRIVATE - PUBLIC) con IC95% ===")
        print(diff_res)
    except Exception as e:
        print(f"[INFO] Comparación de dependencia no ejecutada: {e}")
