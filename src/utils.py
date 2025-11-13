import logging
import yaml
import pandas as pd
import os
import json
import re
import geopandas as gpd

def create_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Crea y configura un logger con consola y archivo opcional.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Evitar duplicar handlers si la función se llama varias veces
    if not logger.handlers:
        # Handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler de archivo, solo si se pasa log_file
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.ERROR)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

class BaseUtils:
    """
    Clase base con métodos utilitarios para cargar parámetros y usar logging.
    """
    def __init__(self, logger: logging.Logger, params_path: str, columns: list = []):
        self.logger = logger
        self.params_path = params_path
        self.columns = columns

    def load_params(self) -> dict:
        """
        Carga un archivo YAML y retorna un diccionario con los parámetros.
        """
        try:
            with open(self.params_path, 'r') as file:
                params = yaml.safe_load(file)
            self.logger.debug('Parameters retrieved from %s', self.params_path)
            return params
        except FileNotFoundError:
            self.logger.error('File not found: %s', self.params_path)
            raise
        except yaml.YAMLError as e:
            self.logger.error('YAML error: %s', e)
            raise
        except Exception as e:
            self.logger.error('Unexpected error: %s', e)
            raise
    
    def load_parquet(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_parquet(path, columns=self.columns)
            self.logger.debug('Parquet file retrived from %s', path)
            return df
        except FileNotFoundError:
            self.logger.error('File not found: %s', path)
            raise
        except Exception as e:
            self.logger.error('Unexpected error: %s', e)
            raise
    
    def downcast_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte automáticamente las columnas numéricas:
        - float64 -> float32
        - int64   -> int32
        """
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')
        return df

    def downcast_numeric_gdf(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Convierte automáticamente las columnas numéricas de un GeoDataFrame:
        - float64 -> float32
        - int64   -> int32
        La columna 'geometry' se mantiene intacta.
        """
        gdf = gdf.copy()  # para no modificar en sitio
        geom_col = gdf.geometry.name

        # Float64 -> float32
        for col in gdf.select_dtypes(include=['float64']).columns:
            if col != geom_col:
                gdf[col] = gdf[col].astype('float32')

        # Int64 -> int32
        for col in gdf.select_dtypes(include=['int64']).columns:
            if col != geom_col:
                gdf[col] = gdf[col].astype('int32')

        return gdf