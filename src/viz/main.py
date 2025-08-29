# src/viz/main.py
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Importamos nuestras funciones del módulo plots
from plots import (
    load_dashboard_csv,
    split_public_private,
    plot_hist_subsidized_public,
    plot_hist_subsidized_private,
    plot_hist_unsubsidized_public,
    plot_hist_unsubsidized_private,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generación de gráficos para dashboard 2010."
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Ruta al CSV (dashboard_2010_clean.csv)."
    )
    parser.add_argument(
        "--outdir", type=str, default="figs", help="Directorio de salida para PNGs."
    )
    args = parser.parse_args()

    # Cargar datos
    df = load_dashboard_csv(args.csv)
    publicas, privadas = split_public_private(df)

    # Directorio de salida
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Generar gráficos
    plot_hist_subsidized_public(
        publicas, savepath=outdir / "hist_subsidized_public.png"
    )
    plt.close()

    plot_hist_subsidized_private(
        privadas, savepath=outdir / "hist_subsidized_private.png"
    )
    plt.close()

    plot_hist_unsubsidized_public(
        publicas, savepath=outdir / "hist_unsubsidized_public.png"
    )
    plt.close()

    plot_hist_unsubsidized_private(
        privadas, savepath=outdir / "hist_unsubsidized_private.png"
    )
    plt.close()


if __name__ == "__main__":
    main()
