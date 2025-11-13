import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import itertools
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from utils import BaseUtils, create_logger

class DataIntegrationClimate(BaseUtils):
    def __init__(self, params_path: str, data_climate_path:str,data_division_path:str , output_dir: str):
        logger = create_logger('data_integration_climate', 'logs/data_integration_climate_errors.log')
        super().__init__(logger=logger,params_path=params_path)
        self.data_climate_path = data_climate_path
        self.data_division_path = data_division_path
        self.output_dir = output_dir
        self.params = self.load_params()["data_integration_climate"]
        self.fecha_cols = None
        self.threshold_m = self.params.get("threshold_m", 20000)
        # ------------------ CRS ------------------
        self.crs_geog = "EPSG:4326"
        self.target_crs_metric = "EPSG:6372"

    def promedio_series_por_municipio(self,df, gdf_muni):
        """
        Convierte puntos a GeoDataFrame, asigna cada estación al municipio de División
        usando CVE_GEO, promedia las series por variable y municipio, y asegura que cada
        municipio tenga 3 filas (Tmax, Tmin, Prec), aunque no haya estaciones.
        """
        try:
            # Crear GeoDataFrame de estaciones
            gdf_points = gpd.GeoDataFrame(
                df.copy(),
                geometry=[Point(xy) for xy in zip(df["Longitud"], df["Latitud"])],
                crs=self.crs_geog
            )

            # Asegurar CRS consistente
            gdf_muni = gdf_muni.to_crs(self.crs_geog)

            # Join espacial: asignar cada estación al municipio de División
            gdf_joined = gpd.sjoin(
                gdf_points,
                gdf_muni[['CVEGEO', 'geometry']],
                how="left",
                predicate="intersects"
            )

            # Convertir series a numéricas
            for col in self.fecha_cols:
                gdf_joined[col] = pd.to_numeric(gdf_joined[col], errors='coerce')

            # Agrupar por municipio (CVEGEO) y variable
            df_agg = gdf_joined.groupby(['CVEGEO', 'Variable'], as_index=False)[self.fecha_cols].mean()

            # Crear todas las combinaciones municipio × variable
            variables = df['Variable'].unique()
            todos_munis_var = pd.DataFrame(
                list(itertools.product(gdf_muni['CVEGEO'].unique(), variables)),
                columns=['CVEGEO', 'Variable']
            )
            self.logger.info(f"✅ Cross product: {len(gdf_muni['CVEGEO'].unique())} municipios × {len(variables)} variables = {len(todos_munis_var)} filas")

            # Merge para asegurar que todos los municipios tengan filas por variable
            df_complete = todos_munis_var.merge(df_agg, on=['CVEGEO', 'Variable'], how='left')

            # Merge con info del municipio (solo una fila por CVE_GEO)
            cols_division = [c for c in gdf_muni.columns if c != 'geometry']
            gdf_muni_unique = gdf_muni[cols_division].drop_duplicates(subset='CVEGEO')

            df_final = pd.merge(
                gdf_muni_unique,
                df_complete,
                on='CVEGEO',
                how='right'
            )

            # Reordenar columnas: división + variable + series
            df_final = df_final[cols_division + ['Variable'] + list(self.fecha_cols)]

            return df_final
        except Exception as e:
            self.logger.error(f"Hubo un error al asignar las series {e}")
    
    def has_any_data(self, row):
        return row[self.fecha_cols].notna().any()
        
    def fill_missing(self, var, gdf_result, coords):
        try:
            mask_var = gdf_result['Variable'] == var
            idx_var = np.where(mask_var)[0]
            if len(idx_var) == 0:
                return None

            mask_has = gdf_result.loc[idx_var].apply(self.has_any_data, axis=1).values
            valid_idx = idx_var[mask_has]
            missing_idx = idx_var[~mask_has]
            if len(valid_idx) == 0 or len(missing_idx) == 0:
                return None

            nbrs = NearestNeighbors(radius=self.threshold_m, algorithm='auto', n_jobs=-1)
            nbrs.fit(coords[valid_idx])
            dists_list, inds_list = nbrs.radius_neighbors(coords[missing_idx])

            filled_values = np.full((len(missing_idx), len(self.fecha_cols)), np.nan, dtype=float)
            for i_miss, inds_rel in enumerate(inds_list):
                if len(inds_rel) == 0:
                    continue
                neighbor_vals = gdf_result.iloc[valid_idx[inds_rel]][self.fecha_cols].values.astype(float)
                col_means = np.nanmean(neighbor_vals, axis=0)
                if not np.all(np.isnan(col_means)):
                    filled_values[i_miss] = col_means

            return missing_idx, filled_values
        except Exception as e:
            self.logger.error(f"Hubo un error rellenando municipios {e}")

    def merging_division_climate(self):
        try:
            # ------------------ Flujo principal ------------------
            df = pd.read_csv(self.data_climate_path)
            gdf_municipios = gpd.read_file(self.data_division_path, encoding='latin-1')
            self.logger.debug(f"{gdf_municipios.shape[0]}")
            self.fecha_cols = df.columns[7:]
            df_merged = self.promedio_series_por_municipio(df, gdf_municipios)
            df_merged = df_merged.drop(columns=['COV_', 'COV_ID'], errors='ignore')
            self.logger.info(f"{df_merged.shape[0]} filas y {df_merged.notna().all(axis=1).sum()/df_merged.shape[0]*100:.2f}% de datos presentes antes de la imputación.")

            # ------------------ Imputación por vecinos paralela ------------------
            fecha_cols = list(self.fecha_cols)
            nombre_col = 'CVEGEO'

            # Centroides métricos
            gdf_muni_cent = gdf_municipios.to_crs(self.target_crs_metric).copy()
            gdf_muni_cent['centroid_geom'] = gdf_muni_cent.geometry.centroid

            # Merge con df_merged
            df_merge = df_merged.merge(gdf_muni_cent, on=nombre_col, how='left')
            gdf_result = gpd.GeoDataFrame(df_merge, geometry='centroid_geom', crs=self.target_crs_metric)

            coords = np.vstack([gdf_result.geometry.x.values, gdf_result.geometry.y.values]).T

            # Ejecutar en paralelo
            results = Parallel(n_jobs=-1)(
                delayed(self.fill_missing)(var, gdf_result, coords)
                for var in gdf_result['Variable'].unique()
            )

            # Aplicar resultados
            for res in results:
                if res is None:
                    continue
                missing_idx, filled_values = res
                gdf_result.iloc[missing_idx, gdf_result.columns.get_indexer(fecha_cols)] = filled_values

            # ------------------ Guardar resultado ------------------
            geom_col_name = gdf_result.geometry.name
            df_merged_filled = pd.DataFrame(gdf_result.drop(columns=[geom_col_name], errors='ignore'))
            del gdf_result

            self.logger.info(f"{df_merged_filled.shape[0]} filas y {df_merged_filled.notna().all(axis=1).sum()/df_merged_filled.shape[0]*100:.2f}% de datos presentes después de la imputación.")

            df_merged_filled.drop(columns=['geometry', 'PERIMETER_x', 'AREA_x'], errors='ignore', inplace=True)
            df_merged_filled.columns = [
                col[:-2] if col.endswith('_x') else col 
                for col in df_merged_filled.columns
            ]
            df_merged_filled.drop(columns=['COV_', 'COV_ID','PERIMETER_y', 'AREA_y','NOM_ENT_y',
                                'NOMGEO_y', 'CVE_MUN_y','CVE_ENT_y'], errors='ignore', inplace=True)
            df_merged_filled = self.downcast_numeric(df_merged_filled)
            output_path = os.path.join(self.output_dir, 'climatologia_merged.parquet')
            df_merged_filled.to_parquet(output_path, index=False)
            print(f"✅ Parquet final guardado: {output_path}")
        except Exception as e:
            self.logger.error(f"Hubo un error al unir division y clima {e}")

def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        csv_clima_path = os.path.join(root_dir, 'data/processed/clima/climatologia_limpio.csv')
        division_path = os.path.join(root_dir, 'data/Division_politica/mun22gw.shp')
        output_dir = os.path.join(root_dir, 'data/processed/clima')

        integration = DataIntegrationClimate(params_path=params_path,
                                            data_climate_path=csv_clima_path,
                                            data_division_path=division_path,
                                            output_dir=output_dir)
        integration.merging_division_climate()
    except Exception as e:
        integration.logger.error(f"Failed to complete the data integration pipeline {e}")
        raise

if __name__ == "__main__":
    main()