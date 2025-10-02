# S4-07 — Preparación para modelado (PIT y marginales)
**Variable analizada:** `FFEL SUBSIDIZED # of Loans Originated`

**Transformaciones y dominios:**
- `x_pos`: datos positivos (x>0) para distribuciones con soporte positivo (Lognormal, Gamma).
- `x_log = log(x_pos)`: para familias sobre la recta real (t-Student; colas pesadas).

**Marginales evaluadas:**
- Lognormal (x>0), Gamma (x>0), t-Student (log x).

**Tabla KS (screening inicial, no comparar entre dominios distintos):**

         marginal domain  KS_stat      p_value                                                      params
  Lognormal (x>0)      x 0.047983 3.271706e-07               (1.8954602848163216, 0.0, 30.563156681665006)
      Gamma (x>0)      x 0.120626 2.229120e-43                (0.3942654346548081, 0.0, 412.9962168304245)
t-Student (log x)  x_log 0.047985 3.266252e-07 (510561525.4992795, 3.4198017677163843, 1.8954445564847662)

**Seleccionada para PIT:** Lognormal (x>0) (dominio: x)

_Nota_: No se han ajustado cópulas aún; este entregable deja lista la base univariada.
