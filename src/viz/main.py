# src/viz/main.py

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Asegúrate de que este script se ejecute desde el nivel del proyecto
# y que el módulo plots esté en el mismo paquete src.viz
from plots import (
    load_dashboard_csv,
    split_public_private,
    plot_hist_subsidized_public,
    plot_hist_subsidized_private,
    plot_hist_unsubsidized_public,
    plot_hist_unsubsidized_private,
    plot_total_subsidized_vs_unsubsidized,
    plot_loan_distribution_by_school_type,
    plot_scatter_recipients_vs_amount,
    plot_top_states_by_subsidized_loans,
    plot_log_hist_unsubsidized_public,
    plot_log_hist_subsidized_private,
    plot_log_hist_subsidized_public,
<<<<<<< HEAD
    plot_institutions_by_state,
    plot_institutions_by_type,
    plot_scatter_subsidized_vs_disbursements,
    plot_correlation_heatmap,
=======
>>>>>>> de7962c (feat: agrega los gráficos para el sprint 4 (sin PDF grande))
    plot_distribution_by_institution_type,
    plot_scatter_log_recipients_vs_originated,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generación de gráficos para dashboard 2010."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=False,
        default="data/clean/dashboard_2010_clean.csv",
        help="Ruta al archivo CSV limpio (dashboard_2010_clean.csv)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="figs",
        help="Directorio de salida para los gráficos",
    )

    args = parser.parse_args()

    # Cargar datos
    df = load_dashboard_csv(args.csv)
    publicas, privadas = split_public_private(df)

    # Crear directorio de salida si no existe
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

    plot_total_subsidized_vs_unsubsidized(
        df, savepath=outdir / "total_subsidized_vs_unsubsidized.png"
    )
    plt.close()

    plot_loan_distribution_by_school_type(
        df, savepath=outdir / "loan_distribution_by_school_type.png"
    )
    plt.close()

    plot_scatter_recipients_vs_amount(
        df, savepath=outdir / "scatter_recipients_vs_amount.png"
    )
    plt.close()

    plot_top_states_by_subsidized_loans(
        df, savepath=outdir / "top_states_subsidized_loans.png"
    )
    plt.close()

    plot_log_hist_unsubsidized_public(
        publicas, savepath=outdir / "hist_log_unsubsidized_public.png"
<<<<<<< HEAD
    )
    plt.close()

    plot_log_hist_subsidized_private(
        privadas, savepath=outdir / "hist_log_subsidized_private.png"
    )
    plt.close()

    plot_log_hist_subsidized_public(
        publicas, savepath=outdir / "hist_log_subsidized_public.png"
    )
    plt.close()

    # Gráfico 2: Distribución por tipo (Gráfico del Sprint)
    plot_distribution_by_institution_type(
        df,
        savepath=outdir / "fig_S4_02_distribution_violin.png",
        variable="$originated",  # o "$disbursed" si se desea
        use_violin=True,  # cambia a False si prefiere boxplot
    )
    plt.close()

    # Gráfico 3: Dispersión log–log con tendencia (Gráfico del Sprint)
    plot_scatter_log_recipients_vs_originated(
        df, savepath=outdir / "fig_S4_03_scatter.png", use_loess=False
    )
    plot_institutions_by_state(df, savepath=outdir / "institutions_by_state.png")
    plt.close()

    plot_institutions_by_type(df, savepath=outdir / "institutions_by_type.png")
    plt.close()

    plot_scatter_subsidized_vs_disbursements(
        df, savepath=outdir / "scatter_subsidized_vs_disbursements.png"
=======
>>>>>>> de7962c (feat: agrega los gráficos para el sprint 4 (sin PDF grande))
    )
    plt.close()

    plot_log_hist_subsidized_private(
        privadas, savepath=outdir / "hist_log_subsidized_private.png"
    )
    plt.close()

    plot_log_hist_subsidized_public(
        publicas, savepath=outdir / "hist_log_subsidized_public.png"
    )
    plt.close()

    # Gráfico 2: Distribución por tipo (Gráfico del Sprint)
    plot_distribution_by_institution_type(
        df,
        savepath=outdir / "fig_S4_02_distribution_violin.png",
        variable="$originated",  # o "$disbursed" si se desea
        use_violin=True,  # cambia a False si prefiere boxplot
    )
    plt.close()

    # Gráfico 3: Dispersión log–log con tendencia (Gráfico del Sprint)
    plot_scatter_log_recipients_vs_originated(
        df, savepath=outdir / "fig_S4_03_scatter.png", use_loess=False
    )
<<<<<<< HEAD
=======

>>>>>>> de7962c (feat: agrega los gráficos para el sprint 4 (sin PDF grande))

if __name__ == "__main__":
    main()
