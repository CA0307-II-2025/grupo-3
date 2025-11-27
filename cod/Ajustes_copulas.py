# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 09:55:37 2025

@author: gsana
"""

# cod/copulas_auto.py (o dentro de cod_copulas.py)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.distributions.copula.api import (
    GaussianCopula,
    StudentTCopula,
    ClaytonCopula,
    GumbelCopula,
    FrankCopula,
)

# ---------------------------------------------------------
# 1. Ajustar varias cópulas y devolver tabla AIC/BIC
# ---------------------------------------------------------


def fit_copula_candidates(u1, u2):
    """
    u1, u2: arrays 1D con PITs (valores en (0,1))

    Devuelve:
      - tabla (DataFrame) con familia, theta, logLik, AIC, BIC
      - info del mejor modelo (dict con 'family', 'theta', 'copula')
    """
    u = np.column_stack([u1, u2])
    n = u.shape[0]

    # Candidatas: puedes comentar/añadir según lo que quieras probar
    families = {
        "gaussian": GaussianCopula,
        "t": StudentTCopula,
        "clayton": ClaytonCopula,
        "gumbel": GumbelCopula,
        "frank": FrankCopula,
    }

    results = []

    for name, Cop in families.items():
        # Crear instancia
        if name == "t":
            # df fijo; si quieres, prueba varios df o haz un loop externo
            cop = Cop(k_dim=2, df=4)
            k_params = 2  # correlación + df
        else:
            cop = Cop(k_dim=2)
            k_params = 1  # solo un parámetro de dependencia

        # Ajustar parámetro de dependencia via Kendall τ
        theta = cop.fit_corr_param(u)  # NO cambia el objeto, solo devuelve el theta

        # Log-verosimilitud de la cópula (suma sobre datos)
        loglik = cop.logpdf(u, args=(theta,)).sum()

        # Criterios de información
        aic = -2 * loglik + 2 * k_params
        bic = -2 * loglik + k_params * np.log(n)

        results.append(
            dict(
                family=name,
                theta=theta,
                logLik=loglik,
                AIC=aic,
                BIC=bic,
                k=k_params,
                copula=cop,
            )
        )

    # Tabla ordenada por AIC
    table = (
        pd.DataFrame([{k: v for k, v in r.items() if k != "copula"} for r in results])
        .set_index("family")
        .sort_values("AIC")
    )

    # Mejor por AIC
    best = min(results, key=lambda r: r["AIC"])

    best_model = {
        "family": best["family"],
        "theta": best["theta"],
        "copula": best["copula"],
    }

    return table, best_model


# ---------------------------------------------------------
# 2. Gráficos de diagnóstico para la cópula elegida
# ---------------------------------------------------------


def plot_copula_diagnostics(u1, u2, copula, theta, n_grid=60):
    """
    u1, u2: datos en (0,1)
    copula: instancia de statsmodels (GaussianCopula, GumbelCopula, etc.)
    theta:  parámetro ajustado (lo que devolvió fit_corr_param)

    Hace:
      - scatter de (u1, u2)
      - contour de la densidad de la cópula
      - QQ-plot de C(u1,u2) vs U(0,1)
    """
    u = np.column_stack([u1, u2])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Scatter de U
    ax = axes[0]
    ax.scatter(u1, u2, alpha=0.4, s=10)
    ax.set_title("Scatter de U1 vs U2")
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")

    # 2) Contour de la densidad de cópula en [0,1]^2
    ax = axes[1]
    grid = np.linspace(1e-4, 1 - 1e-4, n_grid)
    uu, vv = np.meshgrid(grid, grid)
    pts = np.column_stack([uu.ravel(), vv.ravel()])
    z = np.exp(copula.logpdf(pts, args=(theta,))).reshape(uu.shape)

    cs = ax.contour(uu, vv, z, levels=10)
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_title(f"Contorno cópula {type(copula).__name__}")
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_aspect("equal")

    # 3) QQ-plot de C(u1,u2) vs uniforme
    ax = axes[2]
    c_vals = copula.cdf(u, args=(theta,))
    c_vals = np.asarray(c_vals).ravel()
    c_vals.sort()
    n = len(c_vals)
    theo = (np.arange(1, n + 1) - 0.5) / n

    ax.plot(theo, c_vals, ".", ms=4)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_title("QQ-plot de C(U1,U2)")
    ax.set_xlabel("Cuantiles teóricos U(0,1)")
    ax.set_ylabel("Cuantiles empíricos")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 3. Ejemplo de uso con tus PITs
# ---------------------------------------------------------

if __name__ == "__main__":
    # Supón que ya calculaste tus PITs:
    # u1_tilde, u2_tilde = ...

    # Aquí solo pongo un ejemplo falso:
    rng = np.random.default_rng(123)
    u1_tilde = rng.uniform(size=1000)
    u2_tilde = rng.uniform(size=1000)

    table, best = fit_copula_candidates(u1_tilde, u2_tilde)
    print(table)  # tabla AIC/BIC similar a lo que hacías con marginales

    # Gráficos para la mejor cópula
    plot_copula_diagnostics(
        u1_tilde,
        u2_tilde,
        best["copula"],
        best["theta"],
    )
