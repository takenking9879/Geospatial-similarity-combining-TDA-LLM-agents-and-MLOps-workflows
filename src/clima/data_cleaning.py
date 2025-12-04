import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from utils import BaseUtils, create_logger

class DataCleaningClimate(BaseUtils):
    def __init__(self, params_path: str, data_path:str , output_dir: str):
        logger = create_logger('data_cleaning_climate', 'logs/data_cleaning_climate_errors.log')
        super().__init__(logger=logger,params_path=params_path)
        self.data_path = data_path
        self.output_dir = output_dir
        self.params = self.load_params()["data_cleaning_climate"]

    def average_out_n_days(self, df, start_col:int, window:int):
        """
        Promedia cada 'window' columnas de una serie de tiempo.
        
        df: DataFrame original
        start_col: índice de la columna donde inician los días (7 = columnas 0-6 son metadata)
        window: tamaño de la ventana (3 = cada 3 días)
        
        Return:
            nuevo DataFrame con promedios agrupados
        """
        # Copia columnas fijas (metadatos)
        meta = df.iloc[:, :start_col]

        # Extrae la parte de serie de tiempo
        ts = df.iloc[:, start_col:]

        # Número total de columnas de la serie de tiempo
        n = ts.shape[1]

        # Número de bloques de tamaño 'window'
        num_blocks = n // window
        averaged_blocks = []

        for i in range(num_blocks):
            block = ts.iloc[:, i*window : (i+1)*window]
            averaged = block.mean(axis=1)
            averaged_blocks.append(averaged.rename(f"avg_{i+1}"))

        # Concatenar metadata + promedios
        result = pd.concat([meta] + averaged_blocks, axis=1)
        return result

    def cleaning(self):
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            output_path = os.path.join(self.output_dir, 'climatologia_limpio.csv')
            df = pd.read_csv(self.data_path, encoding="utf-8")

            # Columnas de fechas
            fecha_cols = df.columns[7:]
            fecha_cols_limitadas = [f for f in fecha_cols if f <= self.params.get("date_limit", '2024-06-30')]

            # ============================
            # Función para eliminar filas con muchos NaN
            # ============================
            def fila_tiene_many_nans(fila, columnas, umbral=0.40):
                return fila[columnas].isnull().mean() > umbral

            # ============================
            # Filtrado de estaciones con demasiados NaN
            # ============================
            estaciones_a_eliminar = []
            for estacion in df['Estacion'].unique():
                df_est = df[df['Estacion'] == estacion]
                df_relevante = df_est[df_est['Variable'].isin(['TMAX','TMIN','PRECIP'])]
                filas_problematicas = df_relevante.apply(fila_tiene_many_nans, axis=1, columnas=fecha_cols_limitadas, umbral=self.params.get("threshold", 0.4))
                if filas_problematicas.any():
                    estaciones_a_eliminar.append(estacion)
                    
            self.logger.info(f"Estaciones restantes después del filtrado: {df['Estacion'].nunique()}")
            df_limpio = df[~df['Estacion'].isin(estaciones_a_eliminar)]
            df_limpio = df_limpio[df_limpio['Variable'] != 'EVAP']  # eliminar EVAP
            self.logger.info(f"Estaciones restantes después del filtrado: {df_limpio['Estacion'].nunique()}")
            del df_est, df_relevante, filas_problematicas

            df_limpio[fecha_cols_limitadas] = df_limpio[fecha_cols_limitadas].interpolate(method=self.params.get("method", 'linear'), axis=1, limit_direction='both')
            df_limpio = self.downcast_numeric(df_limpio)
            columnas_finales = df.columns[:7].tolist() + fecha_cols_limitadas
            df_limpio = df_limpio[columnas_finales]

            df_limpio = self.average_out_n_days(df_limpio, start_col=7, window=self.params.get('average_n_days', 3))
            df_limpio.to_csv(output_path, index=False)
            self.logger.info(f"Archivo limpio guardado en: {output_path}")
        except Exception as e:
            self.logger.error(f"Hubo un error al limpiar los datos {e}")
            raise
        
def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        data_path = os.path.join(root_dir, 'data/processed/clima/climatologia_todos_los_archivos.csv')
        output_dir = os.path.join(root_dir, 'data/processed/clima/')

        preprocessing = DataCleaningClimate(params_path=params_path, data_path=data_path, output_dir=output_dir)
        preprocessing.cleaning()
    except Exception as e:
        preprocessing.logger.error(f"Failed to complete the data cleaning pipeline {e}")
        raise

if __name__ == "__main__":
    main()