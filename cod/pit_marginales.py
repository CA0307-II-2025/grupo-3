# cod/pit_marginales.py
# S4-07 — Preparación para modelado (PIT y marginales candidatas)
# Usa data/clean/, exporta figuras a figs/, tablas a tablas/ y la nota a docs/notes/
# Ejemplos de uso (desde la raíz del repo):
#   python cod/pit_marginales.py --col loan_amount
#   python cod/pit_marginales.py --csv data/clean/dashboard_2010_clean.csv --col disbursed

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SEED = 123

# ---------------- Rutas del proyecto ----------------
ROOT = Path(__file__).resolve().parents[1]  # raíz del repo (padre de /cod)
DIR_CLEAN = ROOT / "data" / "clean"
DIR_FIGS = ROOT / "figs"
DIR_TABLAS = ROOT / "tablas"
DIR_DOCS = ROOT / "docs"
DIR_NOTES = DIR_DOCS / "notes"

for d in [DIR_FIGS, DIR_TABLAS, DIR_DOCS, DIR_NOTES]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------- Utilidades ----------------
def _sniff_sep(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]
    line = max(text, key=len) if text else ""
    if line.count(";") >= max(line.count(","), line.count("\t")):
        return ";"
    if line.count("\t") > line.count(","):
        return "\t"
    return ","


def _guess_csv_in_clean() -> Path:
    if not DIR_CLEAN.exists():
        sys.exit(f"ERROR: no existe {DIR_CLEAN}. Colocá tu CSV en data/clean/")
    csvs = sorted(DIR_CLEAN.glob("*.csv"))
    if not csvs:
        sys.exit(f"ERROR: no hay archivos .csv en {DIR_CLEAN}.")
    return csvs[0]


def load_data(csv_path: Path | None) -> pd.DataFrame:
    path = csv_path if csv_path else _guess_csv_in_clean()
    sep = _sniff_sep(path)
    df = pd.read_csv(path, sep=sep)
    return df


# Heurística para elegir columna numérica si no se pasa --col
CANDIDATE_NAME_HINTS = [
    "amount",
    "monto",
    "importe",
    "loan",
    "disbursed",
    "originated",
    "valor",
    "pago",
]


def pick_numeric_column(df: pd.DataFrame, user_col: str | None) -> str:
    if user_col:
        if user_col not in df.columns:
            sys.exit(f"ERROR: no existe la columna '{user_col}'.")
        if not pd.api.types.is_numeric_dtype(df[user_col]):
            sys.exit(f"ERROR: '{user_col}' existe pero no es numérica.")
        return user_col
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for hint in CANDIDATE_NAME_HINTS:
        for c in num_cols:
            if hint.lower() in c.lower():
                return c
    if not num_cols:
        sys.exit("ERROR: no hay columnas numéricas en el dataset.")
    return num_cols[0]


# ---------------- Ajustes marginales y PIT ----------------
def fit_candidates(x_pos: np.ndarray, x_log: np.ndarray):
    cands = {}
    ln_params = stats.lognorm.fit(x_pos, floc=0)
    cands["Lognormal (x>0)"] = (stats.lognorm, ln_params, "x")
    gm_params = stats.gamma.fit(x_pos, floc=0)
    cands["Gamma (x>0)"] = (stats.gamma, gm_params, "x")
    t_params = stats.t.fit(x_log)
    cands["t-Student (log x)"] = (stats.t, t_params, "x_log")
    return cands


def ks_table(x_pos: np.ndarray, x_log: np.ndarray, cands: dict) -> pd.DataFrame:
    rows = []
    for name, (dist, params, domain) in cands.items():
        if domain == "x":
            ks_stat, ks_p = stats.kstest(x_pos, dist.cdf, args=params)
        else:
            ks_stat, ks_p = stats.kstest(x_log, dist.cdf, args=params)
        rows.append(
            {
                "marginal": name,
                "domain": domain,
                "KS_stat": float(ks_stat),
                "p_value": float(ks_p),
                "params": tuple(float(p) for p in params),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "KS_stat"], ascending=[True, True])


def plot_marginals_x(x_pos: np.ndarray, cands: dict, out_png: Path, out_pdf: Path):
    xs = np.linspace(x_pos.min(), x_pos.max(), 400)
    plt.figure(figsize=(8, 5))
    plt.hist(
        x_pos, bins=40, density=True, alpha=0.35, edgecolor="black", label="Datos (x>0)"
    )
    for name, (dist, params, domain) in cands.items():
        if domain != "x":
            continue
        pdf = dist.pdf(xs, *params)
        plt.plot(xs, pdf, label=name)
    plt.title("Marginales candidatas (dominio x>0)")
    plt.xlabel("x")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def plot_marginals_xlog(x_log: np.ndarray, cands: dict, out_png: Path, out_pdf: Path):
    xs = np.linspace(x_log.min(), x_log.max(), 400)
    plt.figure(figsize=(8, 5))
    plt.hist(
        x_log,
        bins=40,
        density=True,
        alpha=0.35,
        edgecolor="black",
        label="Datos (log x)",
    )
    for name, (dist, params, domain) in cands.items():
        if domain != "x_log":
            continue
        pdf = dist.pdf(xs, *params)
        plt.plot(xs, pdf, label=name)
    plt.title("Marginal con colas pesadas (dominio log x)")
    plt.xlabel("log(x)")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def plot_pit(
    data: np.ndarray,
    dist,
    params,
    title: str,
    out_png: Path,
    out_pdf: Path,
    domain_label: str,
):
    u = dist.cdf(data, *params)
    u = u[np.isfinite(u)]
    plt.figure(figsize=(7, 4))
    plt.hist(u, bins=20, range=(0, 1), edgecolor="black")
    plt.title(f"PIT bajo {title} (dominio: {domain_label})")
    plt.xlabel("u = F(x)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    return u


def write_note(
    note_path: Path, colname: str, ks_df: pd.DataFrame, best_name: str, best_domain: str
):
    lines = [
        "# S4-07 — Preparación para modelado (PIT y marginales)",
        f"**Variable analizada:** `{colname}`",
        "",
        "**Transformaciones y dominios:**",
        "- `x_pos`: datos positivos (x>0) para distribuciones con soporte positivo (Lognormal, Gamma).",
        "- `x_log = log(x_pos)`: para familias sobre la recta real (t-Student; colas pesadas).",
        "",
        "**Marginales evaluadas:**",
        "- Lognormal (x>0), Gamma (x>0), t-Student (log x).",
        "",
        "**Tabla KS (screening inicial, no comparar entre dominios distintos):**",
        "",
        ks_df.to_string(index=False),
        "",
        f"**Seleccionada para PIT:** {best_name} (dominio: {best_domain})",
        "",
        "_Nota_: No se han ajustado cópulas aún; este entregable deja lista la base univariada.",
    ]
    note_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(
        description="S4-07 PIT y marginales (usa data/clean/)"
    )
    parser.add_argument(
        "--col",
        type=str,
        default=None,
        help="Columna numérica a analizar (p.ej., loan_amount)",
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Ruta CSV específica (opcional)"
    )
    args = parser.parse_args()

    np.random.seed(SEED)

    df = load_data(Path(args.csv) if args.csv else None)
    col = pick_numeric_column(df, args.col)

    x_raw = df[col].astype(float).to_numpy()
    x_raw = x_raw[np.isfinite(x_raw)]
    x_pos = x_raw[x_raw > 0]
    if x_pos.size == 0:
        sys.exit(f"ERROR: la columna '{col}' no tiene valores positivos para ajustar.")

    x_log = np.log(x_pos)
    x_log = x_log[np.isfinite(x_log)]

    cands = fit_candidates(x_pos, x_log)
    ks = ks_table(x_pos, x_log, cands)
    ks_path = DIR_TABLAS / "pit_marginales_summary.csv"
    ks.to_csv(ks_path, index=False)

    plot_marginals_x(
        x_pos,
        cands,
        DIR_FIGS / "marginales_candidatas_x.png",
        DIR_FIGS / "marginales_candidatas_x.pdf",
    )
    plot_marginals_xlog(
        x_log,
        cands,
        DIR_FIGS / "marginales_candidatas_logx.png",
        DIR_FIGS / "marginales_candidatas_logx.pdf",
    )

    ks_x = ks[ks["domain"] == "x"].sort_values("KS_stat")
    ks_log = ks[ks["domain"] == "x_log"].sort_values("KS_stat")
    if not ks_x.empty:
        best_row = ks_x.iloc[0]
    else:
        best_row = ks_log.iloc[0]

    best_name = best_row["marginal"]
    best_domain = best_row["domain"]
    dist, params, _ = cands[best_name]

    if best_domain == "x":
        pit_data = x_pos
        dom_label = "x>0"
    else:
        pit_data = x_log
        dom_label = "log x"

    plot_pit(
        pit_data,
        dist,
        params,
        best_name,
        DIR_FIGS / "pit_best.png",
        DIR_FIGS / "pit_best.pdf",
        dom_label,
    )

    write_note(
        DIR_NOTES / "pit_marginales.md",
        colname=col,
        ks_df=ks,
        best_name=best_name,
        best_domain=best_domain,
    )

    print("✅ S4-07 completado")
    print(f"- CSV usado: {('explícito' if args.csv else 'primero en data/clean/')}")
    print(f"- Columna analizada: {col}")
    print(f"- Mejor marginal (KS dentro de dominio): {best_name}")
    print(f"- Tabla KS: {ks_path}")
    print(
        f"- Figuras: {DIR_FIGS / 'marginales_candidatas_x.png'}, {DIR_FIGS / 'marginales_candidatas_logx.png'}, {DIR_FIGS / 'pit_best.png'}"
    )
    print(f"- Nota: {DIR_NOTES / 'pit_marginales.md'}")


if __name__ == "__main__":
    main()
