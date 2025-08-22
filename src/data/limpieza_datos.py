# Funciones para carga y limpieza de datos
import pandas as pd


class LectorFondosFFEL:
    def __init__(self, ruta_archivo, fila_encabezado=5, filas_encabezado_nivel=2):
        """
        Inicializa el lector de datos FFEL.

        Parámetros:
        - ruta_archivo: str, ruta al archivo Excel (.xlsx)
        - fila_encabezado: int, índice base donde comienzan los encabezados (por defecto 5 → fila 6 en Excel)
        - filas_encabezado_nivel: int, número de filas que conforman el encabezado (por defecto 2)
        """
        self.ruta = ruta_archivo
        self.header_rows = list(
            range(fila_encabezado, fila_encabezado + filas_encabezado_nivel)
        )
        self.df = None

    def cargar_datos(self):
        """Carga el archivo Excel con encabezados multinivel."""
        self.df = pd.read_excel(self.ruta, header=self.header_rows)
        return self.df

    def aplanar_columnas(self, separador="_"):
        """Convierte columnas MultiIndex en nombres planos, ignorando 'Unnamed'."""
        if isinstance(self.df.columns, pd.MultiIndex):

            def limpio(x: object) -> str:
                s = str(x).strip()
                return "" if (s == "" or s.lower().startswith("unnamed")) else s

            self.df.columns = [
                separador.join(
                    [lvl for lvl in (limpio(level) for level in col) if lvl != ""]
                )
                for col in self.df.columns
            ]
        else:
            # Si algunas columnas simples venían como 'Unnamed: 0'
            self.df.columns = [
                "" if (str(c).lower().startswith("unnamed")) else str(c)
                for c in self.df.columns
            ]
        return self.df

    def limpiar_datos(self):
        """Elimina filas vacías completamente y reinicia índices."""
        self.df.dropna(how="all", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def convertir_numericos(self):
        """Intenta convertir todas las columnas a tipo numérico cuando sea posible."""
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_numeric(
                    self.df[col].astype(str).str.replace(r"[\$,]", "", regex=True),
                    errors="ignore",
                )
            except Exception:
                pass
        return self.df

    def eliminar_filas_por_na(self, umbral=0.5):
        """
        Elimina filas que tengan un porcentaje de valores NA superior al umbral.

        Parámetros:
        - umbral: float, entre 0 y 1.
        Ejemplo: umbral=0.5 elimina filas con más del 50% de valores NA.

        Retorna:
        - DataFrame sin las filas eliminadas.
        """
        if self.df is None:
            raise ValueError("No hay DataFrame cargado. Use cargar_datos primero.")

        # Calcula el porcentaje de NAs por fila
        porcentaje_na = self.df.isna().mean(axis=1)

        # Filtra filas donde el porcentaje de NAs sea menor o igual al umbral
        self.df = self.df.loc[porcentaje_na <= umbral].copy()

        return self.df

    def cargar_y_limpiar(self):
        """Proceso completo de carga y limpieza."""
        self.cargar_datos()
        self.aplanar_columnas()
        self.eliminar_filas_por_na()
        self.limpiar_datos()
        self.convertir_numericos()
        return self.df
