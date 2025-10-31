# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 16:43:03 2025

@author: jeike
"""

import os, re
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.stats import skew, kurtosis, burr12, fisk, genextreme, pareto, genpareto
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from numpy.linalg import cholesky
from scipy.special import gammaln
from scipy.stats import norm, t as student_t


def resumen_disbursements(df_filtrado, tipo_col = 'School Type'):
    """
    Calcula asimetría, curtosis y recomendación de transformación logarítmica
    para todas las variables que contengan 'Disbursements', separadas por tipo de universidad.
    Devuelve un DataFrame resumen.
    """
    cols_disb = [c for c in df_filtrado.columns if 'Disbursements' in c]
    tipos = df_filtrado[tipo_col].dropna().unique()

    registros = []

    for col in cols_disb:
        for tipo in tipos:
            data = df_filtrado[df_filtrado[tipo_col] == tipo][col].dropna()

            # Cálculo en escala original
            s_orig = skew(data)
            k_orig = kurtosis(data)

            # Escala log(1+x)
            data_log = np.log1p(data)
            s_log = skew(data_log)
            k_log = kurtosis(data_log)

            # Regla de decisión: preferir log si reduce skew y kurtosis
            mejor_log = (abs(s_log) < abs(s_orig)) and (abs(k_log - 3) < abs(k_orig - 3))
            recomendacion = "usar log" if mejor_log else "mantener original"

            registros.append({
                "variable": col,
                "tipo_universidad": tipo,
                "skew_original": round(s_orig, 3),
                "kurt_original": round(k_orig, 3),
                "skew_log": round(s_log, 3),
                "kurt_log": round(k_log, 3),
                "recomendacion": recomendacion
            })

    resumen = pd.DataFrame(registros)
    return resumen.sort_values(["variable", "tipo_universidad"]).reset_index(drop=True)



def resumen_disbursements(df_filtrado, tipo_col = 'School Type'):
    """
    Calcula asimetría, curtosis y recomendación de transformación logarítmica
    para todas las variables que contengan 'Disbursements', separadas por tipo de universidad.
    Devuelve un DataFrame resumen.
    """
    cols_disb = [c for c in df_filtrado.columns if 'Disbursements' in c]
    tipos = df_filtrado[tipo_col].dropna().unique()

    registros = []

    for col in cols_disb:
        for tipo in tipos:
            data = df_filtrado[df_filtrado[tipo_col] == tipo][col].dropna()

            # Cálculo en escala original
            s_orig = skew(data)
            k_orig = kurtosis(data)

            # Escala log(1+x)
            data_log = np.log1p(data)
            s_log = skew(data_log)
            k_log = kurtosis(data_log)

            # Regla de decisión: preferir log si reduce skew y kurtosis
            mejor_log = (abs(s_log) < abs(s_orig)) and (abs(k_log - 3) < abs(k_orig - 3))
            recomendacion = "usar log" if mejor_log else "mantener original"

            registros.append({
                "variable": col,
                "tipo_universidad": tipo,
                "skew_original": round(s_orig, 3),
                "kurt_original": round(k_orig, 3),
                "skew_log": round(s_log, 3),
                "kurt_log": round(k_log, 3),
                "recomendacion": recomendacion
            })

    resumen = pd.DataFrame(registros)
    return resumen.sort_values(["variable", "tipo_universidad"]).reset_index(drop=True)



def apply_log(df_filtrado):
    cols_disb = [c for c in df_filtrado.columns if "Disbursements" in c]
    df_log = df_filtrado.copy()
    df_log[cols_disb] = np.log1p(df_filtrado[cols_disb])
    
    return df_log



def fit_distributions(series, distribs = None):
    """
    Ajusta varias distribuciones a una variable y devuelve métricas de bondad de ajuste.
    """
    if distribs is None:
        distribs = ["lognorm", "gamma", "expon", "norm", "t", "weibull_min", "skewnorm", "johnsonsu", "laplace", "pareto", "genpareto", "burr12", "fisk", "genextreme"]

    data = series.dropna().values
    results = []

    for name in distribs:
        dist = getattr(st, name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = dist.fit(data)
        loglik = np.sum(dist.logpdf(data, *params))
        k = len(params)
        n = len(data)
        aic = 2*k - 2*loglik
        bic = k*np.log(n) - 2*loglik
        ks_stat, ks_p = st.kstest(data, name, args=params)
        results.append({
            'distribution': name,
            'params': params,
            'AIC': aic,
            'BIC': bic,
            'KS_stat': ks_stat,
            'KS_p': ks_p
        })
    df_results = pd.DataFrame(results).sort_values('AIC').reset_index(drop=True)
    return df_results



def resumen_fit_distributions(df_log, col_name, school_type):
    """
    Devuelve un DataFrame con las mejores distribuciones (por AIC) para una variable.
    """
    
    df = df_log[df_log['School Type'].str.lower().isin([school_type])]
    serie = df[col_name].dropna()
    ajuste = fit_distributions(serie)

    filas = []
    
    for i in range(10):
        fila = ajuste.iloc[i]
        filas.append({
            'variable': col_name,
            'distribution': fila['distribution'],
            'params': fila['params'],
            'AIC': fila['AIC'],
            'BIC': fila['BIC'],
            'KS_stat': fila['KS_stat'],
            'KS_p': fila['KS_p']
        })

    resumen = pd.DataFrame(filas)
    return resumen



def plot_best_fit_distribution(df_log, col_name, school_type):
    """
    Grafica los datos junto con la mejor distribución ajustada según AIC.
    """
    df = df_log[df_log['School Type'].str.lower().isin([school_type])]
    serie = df[col_name].dropna().values
    ajuste = fit_distributions(pd.Series(serie))
    mejor = ajuste.iloc[0]

    dist_name = mejor['distribution']
    params = mejor['params']
    dist = getattr(st, dist_name)

    x = np.linspace(min(serie), max(serie), 300)
    pdf_teo = dist.pdf(x, *params)

    plt.figure(figsize=(8,5))
    plt.hist(serie, bins='auto', density=True, alpha=0.6, color='lightsteelblue', label='Datos')
    plt.plot(x, pdf_teo, 'r-', lw=2, label=f'{dist_name} fit')

    plt.title(
        f"{col_name}\nMejor ajuste: {dist_name + school_type} | AIC={mejor['AIC']:.2f} | KS={mejor['KS_stat']:.3f}",
        fontsize=12, fontweight='bold'
    )
    plt.xlabel("Valores (log1p)")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mejor


def apply_pit(series, dist_name, params):
    """
    Aplica el Probability Integral Transform (PIT) a una serie de datos
    usando la CDF de la distribución ajustada.
    """
    series = series.dropna()
    dist = getattr(st, dist_name)

    u = dist.cdf(series, *params)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    
    return pd.Series(u, index=series.index, name=f"{series.name}_PIT")


def school_type_pit(df_log, school_type):
    df = df_log[df_log["School Type"] == school_type.upper()].copy()
    cols_disb = [c for c in df_filtrado.columns if "Disbursements" in c]
    
    for col in cols_disb:
        dist = resumen_fit_distributions(df, col, school_type)
        dist_name = dist["distribution"][0]
        params = dist["params"][0]

        df[f"{col}_PIT"] = apply_pit(df[col], dist_name, params)
    
    return df


def diagnosticar_pit(df, col_name):
    """
    Genera diagnóstico visual del PIT:
    - Histograma del PIT
    - CDF empírica vs teórica (U(0,1))
    - QQ-plot del PIT vs uniforme
    """
    u = df[col_name].dropna().values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(u, bins="auto", density=True, color='lightsteelblue', edgecolor='gray', alpha=0.7)
    axes[0].axhline(1, color='red', linestyle='--', label='Uniforme ideal')
    axes[0].set_title("Histograma del PIT", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("u_i")
    axes[0].set_ylabel("Densidad")
    axes[0].legend()
    
    u_sorted = np.sort(u)
    cdf_emp = np.arange(1, len(u_sorted)+1) / len(u_sorted)
    cdf_teo = u_sorted 
    axes[1].plot(u_sorted, cdf_emp, label="CDF empírica", color="steelblue")
    axes[1].plot(u_sorted, cdf_teo, label="CDF teórica U(0,1)", color="red", linestyle="--")
    axes[1].fill_between(u_sorted, cdf_emp, cdf_teo, color="gray", alpha=0.2)
    axes[1].set_title("CDF empírica vs teórica", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("u_i")
    axes[1].set_ylabel("Probabilidad acumulada")
    axes[1].legend()

    st.probplot(u, dist="uniform", plot=axes[2])
    axes[2].get_lines()[1].set_color('red') 
    axes[2].set_title("QQ-plot vs Uniforme(0,1)", fontsize=11, fontweight='bold')

    fig.suptitle(f"Diagnóstico del PIT — {col_name}", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"../figs/diagnostico_pit_{col_name.replace(' ', '_').replace('$','').replace('/', '-')}.png", dpi=300)
    plt.show()






# ---------- utilidades ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def slug(s: str):
    s = s.replace('$', 'USD').replace('#', 'N')   # distingue $ vs #
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def build_U(df_public, df_private, df_proprietary, pit_col):
    U = pd.concat([
        df_public[pit_col].reset_index(drop=True),
        df_private[pit_col].reset_index(drop=True),
        df_proprietary[pit_col].reset_index(drop=True)
    ], axis=1).dropna()
    U.columns = ["Public","Private","Proprietary"]
    return U

# ---------- Gauss copula ----------
def fit_gaussian_copula(U_df):
    U = U_df.values
    Z = st.norm.ppf(U)
    R = np.corrcoef(Z, rowvar=False)
    R = 0.999*R + 0.001*np.eye(3)  # SPD firme

    logpdf_mvn = multivariate_normal(mean=np.zeros(3), cov=R).logpdf(Z)
    logpdf_std = st.norm.logpdf(Z).sum(axis=1)
    ll = float(np.sum(logpdf_mvn - logpdf_std))

    k, n = 3, len(U_df)  # 3 off-diagonales
    aic = -2*ll + 2*k
    bic = -2*ll + k*np.log(n)

    # simulación
    L = np.linalg.cholesky(R)
    Zsim = np.random.randn(n,3) @ L.T
    Usim = st.norm.cdf(Zsim)
    sim = pd.DataFrame(Usim, columns=U_df.columns)
    return {"rho": R}, ll, aic, bic, sim

# ---------- t copula ----------
def _logpdf_mvt(T, R, nu):
    d = T.shape[1]
    invR = np.linalg.inv(R)
    sign, logdet = np.linalg.slogdet(R)
    quad = np.einsum('ij,jk,ik->i', T, invR, T)
    c = (gammaln((nu+d)/2) - gammaln(nu/2) - 0.5*logdet - (d/2)*np.log(nu*np.pi))
    return c - 0.5*(nu+d)*np.log1p(quad/nu)

def fit_t_copula(U_df, nus=(4,6,8,10,15,20,30,60)):
    U = U_df.values
    n, d = len(U_df), U_df.shape[1]
    best = None
    for nu in nus:
        T = st.t.ppf(U, df=nu)
        R = np.corrcoef(T, rowvar=False)
        R = 0.999*R + 0.001*np.eye(d)
        ll = float(np.sum(_logpdf_mvt(T, R, nu) - st.t.logpdf(T, df=nu).sum(axis=1)))
        k = 3 + 1
        aic = -2*ll + 2*k
        bic = -2*ll + k*np.log(n)
        if (best is None) or (aic < best["AIC"]):
            best = {"nu": nu, "R": R, "LogLik": ll, "AIC": aic, "BIC": bic}
    # simulación con el mejor
    nu, R = best["nu"], best["R"]
    L = np.linalg.cholesky(R)
    Z = np.random.randn(n,d) @ L.T
    g = np.random.chisquare(df=nu, size=n)
    Tsim = Z/np.sqrt(g[:,None]/nu)
    Usim = st.t.cdf(Tsim, df=nu)
    sim = pd.DataFrame(Usim, columns=U_df.columns)
    return {"rho": R, "nu": nu}, best["LogLik"], best["AIC"], best["BIC"], sim

# ---------- métricas de dependencia ----------
def tau_kendall(U_df):
    return U_df.corr(method="kendall")

def tail_dep_t(rho, nu):
    # λ_U = 2 * t_{ν+1}\left( - sqrt((ν+1)*(1-ρ)/(1+ρ)) \right)
    x = -np.sqrt((nu+1)*(1-rho)/(1+rho))
    lam = 2*st.t.cdf(x, df=nu+1)
    return lam  # λ_L = λ_U por simetría en t

def joint_probs(sim_df, q=0.95):
    U = sim_df.values
    P12 = np.mean((U[:,0]>q)&(U[:,1]>q))
    P13 = np.mean((U[:,0]>q)&(U[:,2]>q))
    P23 = np.mean((U[:,1]>q)&(U[:,2]>q))
    P123= np.mean((U[:,0]>q)&(U[:,1]>q)&(U[:,2]>q))
    return {"P12":P12,"P13":P13,"P23":P23,"P123":P123}

# ---------- gráficos ----------
def plot_tau_heatmap(tau, outpng):
    plt.figure(figsize=(4,3.5))
    sns.heatmap(tau, vmin=-1, vmax=1, annot=True, fmt=".2f", square=True, cbar=True)
    plt.title("Kendall τ (U, U)")
    plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()
    
    
def kendall_tau_from_copula(rho, columns):
    """τ de Kendall implícita por la cópula (Gauss/t): tau = (2/π) * arcsin(rho)."""
    R = np.asarray(rho, dtype=float)
    R = np.clip(R, -0.999999, 0.999999)
    T = (2/np.pi) * np.arcsin(R)
    np.fill_diagonal(T, 1.0)
    return pd.DataFrame(T, index=columns, columns=columns)

def plot_emp_vs_sim(U_df, sim_df, outpng):
    df = pd.concat([U_df.add_suffix("_emp"), sim_df.add_suffix("_sim")], axis=1)
    fig, axs = plt.subplots(1,3, figsize=(11,3.5))
    pairs = [("Public","Private"),("Public","Proprietary"),("Private","Proprietary")]
    for ax,(i,j) in zip(axs,pairs):
        ax.scatter(U_df[i], U_df[j], s=6, alpha=0.35, label="Empírico")
        ax.scatter(sim_df[i], sim_df[j], s=6, alpha=0.35, label="Simulado")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel(i); ax.set_ylabel(j)
    axs[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("U empírico vs simulado (modelo ganador)", y=1.02)
    plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()

def gaussian_copula_logpdf(U, R):
    z = st.norm.ppf(U)
    invR = np.linalg.inv(R); sign, logdet = np.linalg.slogdet(R)
    Q = np.einsum("ij,jk,ik->i", z, invR - np.eye(R.shape[0]), z)
    return -0.5*logdet - 0.5*Q

def t_copula_logpdf(U, R, nu):
    d = R.shape[0]
    z = st.t.ppf(U, df=nu)
    invR = np.linalg.inv(R); sign, logdet = np.linalg.slogdet(R)
    quad = np.einsum("ij,jk,ik->i", z, invR, z)
    log_mt = (gammaln((nu+d)/2) - gammaln(nu/2)
              - 0.5*logdet - (d/2)*np.log(nu*np.pi)
              - 0.5*(nu+d)*np.log1p(quad/nu))
    log_uni = st.t.logpdf(z, df=nu).sum(axis=1)
    return log_mt - log_uni

def plot_copula_contour(model, params, outpng, u_fix=0.5, dims=(0,1), grid=200):
    # slice: c(u_i, u_j | u_k = 0.5)
    i,j = dims
    d = 3
    U = np.zeros((grid*grid, d))
    u1 = np.linspace(1e-6, 1-1e-6, grid)
    u2 = np.linspace(1e-6, 1-1e-6, grid)
    U[:,i] = np.repeat(u1, grid)
    U[:,j] = np.tile(u2, grid)
    k = [0,1,2]; k.remove(i); k.remove(j)
    U[:,k[0]] = u_fix

    if model == "Gaussian":
        logc = gaussian_copula_logpdf(U, params["rho"])
    else:
        logc = t_copula_logpdf(U, params["rho"], params["nu"])
    C = np.exp(logc).reshape(grid, grid)

    plt.figure(figsize=(5,4))
    cs = plt.contourf(u1, u2, C.T, levels=20)
    plt.colorbar(cs, shrink=.85, label="c(u)")
    names = ["Public","Private","Proprietary"]
    plt.xlabel(names[i]); plt.ylabel(names[j])
    plt.title(f"Densidad {model} — slice {names[k[0]]} = 0.5")
    plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()

# ---------- orquestador ----------
def generate_copula_report(df_public, df_private, df_proprietary,
                           outdir="results/copulas", q_extreme=0.95):
    ensure_dir(outdir)

    pits_pub  = {c for c in df_public.columns if c.endswith("_PIT")}
    pits_priv = {c for c in df_private.columns if c.endswith("_PIT")}
    pits_prop = {c for c in df_proprietary.columns if c.endswith("_PIT")}
    pit_cols = sorted(list(pits_pub & pits_priv & pits_prop))
    if not pit_cols:
        raise ValueError("No encontré columnas '_PIT' comunes en los tres dataframes.")

    summary_rows = []

    for col in pit_cols:
        print(f"\n Procesando: {col}")
        U = build_U(df_public, df_private, df_proprietary, col)

        # Ajustes
        g_params, g_ll, g_aic, g_bic, g_sim = fit_gaussian_copula(U)
        t_params, t_ll, t_aic, t_bic, t_sim = fit_t_copula(U)

        # Ganador por AIC
        if t_aic < g_aic:
            winner = "t"; params = t_params; ll=t_ll; aic=t_aic; bic=t_bic; sim=t_sim
        else:
            winner = "Gaussian"; params = g_params; ll=g_ll; aic=g_aic; bic=g_bic; sim=g_sim

        # τ de Kendall de la CÓPULA (no empírica)
        tau_mod = kendall_tau_from_copula(params["rho"], U.columns)

        # Carpeta
        var_slug = slug(col)
        vdir = os.path.join(outdir, var_slug); ensure_dir(vdir)

        # Tabla A
        tableA = pd.DataFrame([{
            "Variable": col, "Ganador": winner, "LogLik": ll, "AIC": aic, "BIC": bic,
            "rho12": params["rho"][0,1], "rho13": params["rho"][0,2], "rho23": params["rho"][1,2],
            **({"nu": params["nu"]} if winner=="t" else {})
        }])
        tableA.to_csv(os.path.join(vdir, "tabla_modelo.csv"), index=False)
        summary_rows.append(tableA.iloc[0].to_dict())

        # Dependencia de cola (solo t)
        if winner == "t":
            lam12 = tail_dep_t(params["rho"][0,1], params["nu"])
            lam13 = tail_dep_t(params["rho"][0,2], params["nu"])
            lam23 = tail_dep_t(params["rho"][1,2], params["nu"])
        else:
            lam12 = lam13 = lam23 = 0.0

        # Probabilidades conjuntas desde simulados del ganador
        jp = joint_probs(sim, q=q_extreme)

        # --- TABLA B: usa τ de la CÓPULA ---
        tableB = pd.DataFrame([
            {"Par":"Public–Private",     "tau_copula":tau_mod.loc["Public","Private"],      "rho_model":params["rho"][0,1], "lambda_L":lam12, "lambda_U":lam12},
            {"Par":"Public–Proprietary", "tau_copula":tau_mod.loc["Public","Proprietary"],  "rho_model":params["rho"][0,2], "lambda_L":lam13, "lambda_U":lam13},
            {"Par":"Private–Proprietary","tau_copula":tau_mod.loc["Private","Proprietary"], "rho_model":params["rho"][1,2], "lambda_L":lam23, "lambda_U":lam23},
            {"Par":"P(>q) conj. pares",  "tau_copula":np.nan, "rho_model":np.nan, "lambda_L":np.nan, "lambda_U":np.nan,
             **{"P12":jp["P12"],"P13":jp["P13"],"P23":jp["P23"],"P123":jp["P123"]}}
        ])
        tableB.to_csv(os.path.join(vdir, "tabla_dependencia.csv"), index=False)

        # Figuras: heatmap de τ de la CÓPULA + scatter empírico vs simulado + contorno
        plot_tau_heatmap(tau_mod, os.path.join(vdir, "fig1_tau_copula_heatmap.png"))
        plot_emp_vs_sim(U, sim, os.path.join(vdir, "fig2_emp_vs_sim.png"))

        # par más dependiente según |τ de la CÓPULA|
        abs_tau = tau_mod.replace(1.0, np.nan).abs()
        pair_idx = np.unravel_index(np.nanargmax(abs_tau.values), abs_tau.values.shape)
        dims_map = {("Public","Private"):(0,1), ("Public","Proprietary"):(0,2), ("Private","Proprietary"):(1,2)}
        top_pair = (tau_mod.index[pair_idx[0]], tau_mod.columns[pair_idx[1]])
        dims = dims_map.get(top_pair, (0,1))
        plot_copula_contour(winner, params, os.path.join(vdir, "fig3_contour_top_pair.png"),
                            u_fix=0.5, dims=dims, grid=120)

    # Resumen global
    summary = pd.DataFrame(summary_rows).sort_values(["AIC","Variable"])
    summary.to_csv(os.path.join(outdir, "_summary_models.csv"), index=False)
    print(f"\n Listo. Archivos en: {outdir}")




df = pd.read_csv("../data/clean/dashboard_2010_clean.csv")

tipos = ['public', 'private', 'proprietary']

df_filtrado = df[df['School Type'].str.lower().isin(tipos)]

df_log = apply_log(df_filtrado)

df_public = school_type_pit(df_log, "public")
df_private = school_type_pit(df_log, "private")
df_proprietary = school_type_pit(df_log, "proprietary")

generate_copula_report(df_public, df_private, df_proprietary,
                           outdir="results/copulas", q_extreme=0.95)