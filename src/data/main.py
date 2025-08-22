import os
from limpieza_datos import LectorFondosFFEL

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RUTA_RAW = os.path.join(ROOT_DIR, "data", "raw", "FL_Dashboard_AY2009_2010_Q4.xlsx")
RUTA_CLEAN = os.path.join(ROOT_DIR, "data", "clean", "dashboard_2010_clean.csv")

# Código para cargar y limpiar datos de fondos FFEL
lector = LectorFondosFFEL(
    ruta_archivo=RUTA_RAW, fila_encabezado=4, filas_encabezado_nivel=2
)
df = lector.cargar_datos()
df = lector.aplanar_columnas(separador=" ")
df = lector.limpiar_datos()
df = lector.convertir_numericos()
df = lector.eliminar_filas_por_na(umbral=0.7)

print("✅ Datos cargados y limpiados correctamente")
print(df.head())

df.to_csv(RUTA_CLEAN, index=False)
print(f"📁 Archivo limpio guardado en: {RUTA_CLEAN}")
