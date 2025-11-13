import geopandas as gpd
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from utils import BaseUtils, create_logger
from functools import reduce

class DataPreprocessingSoil(BaseUtils):
    def __init__(self, params_path: str, division_dir:str , raw_eda_dir: str, raw_soil_dir:str , preprocessed_data_dir: str = None):
        logger = create_logger('data_preprocessing_soil', 'logs/data_preprocessing_soil_errors.log')
        super().__init__(logger=logger,params_path=params_path)
        self.raw_soil_dir = raw_soil_dir
        self.raw_eda_dir = raw_eda_dir
        self.division_dir = division_dir
        self.preprocessed_data_dir = preprocessed_data_dir or os.path.join("data", "processed_soil")
        self.params = self.load_params()["data_preprocessing_soil"]

    def load_files(self):
        try:
            gdf_division = gpd.read_file(self.division_dir, encoding='latin-1')
            gdf_eda = gpd.read_file(self.raw_eda_dir)
            gdf_suelo = gpd.read_file(self.raw_soil_dir)

            # Eliminamos columnas que no serán utiles
            gdf_eda = gdf_eda.drop(columns=['ID_FOTO', 'CAL_POS'])
            # Eliminamos filas con capas del suelo que no afectan al cultivo
            gdf_eda = gdf_eda[~gdf_eda['NOMEN_HTE'].isin(['CR1','CR2'])]
            gdf_eda = gdf_eda.set_crs("EPSG:6372", allow_override=True)
            gdf_division= gdf_division.to_crs(epsg=6372)
            gdf_suelo = gdf_suelo.to_crs(gdf_division.crs)

            self.logger.debug(f"El crs de division es: {gdf_division.crs}")
            self.logger.debug(f"El crs de edafologia es: {gdf_eda.crs}")
            self.logger.debug(f"El crs de uso de suelo es: {gdf_suelo.crs}")

            return gdf_division, gdf_eda, gdf_suelo
        except Exception as e:
            self.logger.error(f"Hubo un error al cargar los datos {e}")

    def merge_categories(self, gdf_suelo: pd.DataFrame) -> pd.DataFrame: 
        """ Uniendo Suelo y división"""

        gdf_suelo['DESCRIPCIO'] = gdf_suelo['DESCRIPCIO'].apply(
            lambda x: 'VEGETACIÓN SECUNDARIA HERBÁCEA' if str(x).startswith('VEGETACIÓN SECUNDARIA HERBÁCEA') or str(x).startswith('VEGETACION SECUNDARIA HERBACEA') else x
        )
        gdf_suelo['DESCRIPCIO'] = gdf_suelo['DESCRIPCIO'].apply(
            lambda x: 'VEGETACIÓN SECUNDARIA ARBUSTIVA' if str(x).startswith('VEGETACIÓN SECUNDARIA ARBUSTIVA') or str(x).startswith('VEGETACION SECUNDARIA ARBUSTIVA') else x
        )
        gdf_suelo['DESCRIPCIO'] = gdf_suelo['DESCRIPCIO'].apply(
            lambda x: 'VEGETACIÓN SECUNDARIA ARBÓREA' if str(x).startswith('VEGETACIÓN SECUNDARIA ARBÓREA') else x
        )
        gdf_suelo['DESCRIPCIO'] = gdf_suelo['DESCRIPCIO'].apply(
            lambda x: 'SELVA BAJA' if str(x).startswith('SELVA BAJA') else x
        )
        # Diccionario de mapeo: clase original -> categoría
        categoria_vegetacion = {
            # Bosques
            'BOSQUE DE OYAMEL': 'BOSQUE OYAMEL',
            'BOSQUE DE CEDRO': 'BOSQUE CEDRO',
            'BOSQUE CULTIVADO': 'BOSQUE CULTIVADO',
            'BOSQUE DE GALERÍA': 'BOSQUE GALERIA',
            'BOSQUE INDUCIDO': 'BOSQUE INDUCIDO',
            'BOSQUE DE TÁSCATE': 'BOSQUE TASCATE',
            'BOSQUE MESÓFILO DE MONTAÑA': 'BOSQUE MESOFILO',
            'BOSQUE DE PINO': 'BOSQUE PINO',
            'BOSQUE DE PINO-ENCINO': 'BOSQUE PINO-ENCINO',
            'BOSQUE DE ENCINO': 'BOSQUE ENCINO',
            'BOSQUE DE ENCINO-PINO': 'BOSQUE PINO-ENCINO',
            'BOSQUE DE AYARÍN': 'BOSQUE AYARIN',
            'BOSQUE DE MEZQUITE': 'BOSQUE MEZQUITE',

            # Matorrales y chaparral
            'MATORRAL CRASICAULE': 'MATORRAL CRASICAULE',
            'MATORRAL DESÉRTICO MICRÓFILO': 'MATORRAL DESERTICO MICROFILO',
            'MATORRAL DESÉRTICO ROSETÓFILO': 'MATORRAL DESERTICO ROSETÓFILO',
            'MATORRAL ESPINOSO TAMAULIPECO': 'MATORRAL ESPINOSO',
            'MATORRAL ROSETÓFILO COSTERO': 'MATORRAL COSTERO',
            'MATORRAL SARCOCAULE': 'MATORRAL SARCOCAULE',
            'MATORRAL SARCO-CRASICAULE': 'MATORRAL SARCO CRASICAULE',
            'MATORRAL SUBMONTANO': 'MATORRAL SUBMONTANO',
            'MATORRAL SARCO-CRASICAULE DE NEBLINA': 'MATORRAL NEBLINA',
            'MATORRAL SUBTROPICAL': 'MATORRAL SUBTROPICAL',
            'CHAPARRAL': 'CHAPARRAL',

            # Pastizales y praderas
            'PASTIZAL CULTIVADO': 'PASTIZAL CULTIVADO',
            'PASTIZAL HALÓFILO': 'PASTIZAL HALOFILO',
            'PASTIZAL INDUCIDO': 'PASTIZAL INDUCIDO',
            'PASTIZAL NATURAL': 'PASTIZAL NATURAL',
            'PASTIZAL GIPSÓFILO': 'PASTIZAL GIPSOFILO',
            'PRADERA DE ALTA MONTAÑA': 'PRADERA ALTA MONTAÑA',
            'SABANA': 'SABANA',
            'SABANOIDE': 'SABANOIDE',

            # Humedales / cuerpos de agua
            'VEGETACIÓN DE PETEN': 'VEGETACION DE PETEN',

            # Manglares y palmares
            'MANGLAR': 'MANGLAR',
            'PALMAR INDUCIDO': 'PALMAR INDUCIDO',
            'PALMAR NATURAL': 'PALMAR NATURAL',

            # Vegetación secundaria
            'VEGETACIÓN SECUNDARIA ARBUSTIVA': 'VEGETACION SECUNDARIA ARBUSTIVA',
            'VEGETACIÓN SECUNDARIA ARBÓREA': 'VEGETACION SECUNDARIA ARBOREA',
            'VEGETACIÓN SECUNDARIA HERBÁCEA': 'VEGETACION SECUNDARIA HERBACEA',

            # Otros / sin vegetación
            'ACUÍCOLA': 'ACUICOLA',
            'DESPROVISTO DE VEGETACIÓN': 'SIN VEGETACION',
            'SIN VEGETACIÓN APARENTE': 'SIN VEGETACION',
            'ASENTAMIENTOS HUMANOS': 'ASENTAMIENTOS',
            'VEGETACIÓN DE DESIERTOS ARENOSOS': 'VEGETACION DESIERTO ARENOSO',
            'VEGETACIÓN DE GALERÍA': 'VEGETACION GALERIA',
            'VEGETACIÓN HALÓFILA XERÓFILA': 'VEGETACION HALOFILA XEROFILA',
            'VEGETACIÓN HALÓFILA HIDRÓFILA': 'VEGETACION HALOFILA HIDROFILA',
            'VEGETACIÓN DE DUNAS COSTERAS': 'VEGETACION DUNAS',
            'VEGETACIÓN GIPSÓFILA': 'VEGETACION GIPSOFILA'
        }
        gdf_suelo['DESCRIPCIO'] = gdf_suelo['DESCRIPCIO'].map(categoria_vegetacion)
        return gdf_suelo
    
    def merge_division_soil(self) -> None:
        try:
            os.makedirs(self.preprocessed_data_dir, exist_ok=True)  # exist_ok=True evita error si ya existe
            gdf_division, gdf_eda, gdf_suelo = self.load_files()
            gdf_suelo = self.merge_categories(gdf_suelo)
            # --- 1. Intersección real (evita doble conteo) ---
            intersections = gpd.overlay(gdf_division, gdf_suelo, how='intersection')

            # --- 2. Calcular área de la intersección ---
            intersections['area_intersection'] = intersections.geometry.area

            # --- 3. Agrupar por polígono y categoría ---
            area_by_type = intersections.groupby(['CVEGEO', 'DESCRIPCIO'])['area_intersection'].sum().reset_index()

            # --- 4. Pivotear para tener columna por categoría ---
            pivot_area = area_by_type.pivot(index='CVEGEO', columns='DESCRIPCIO', values='area_intersection').fillna(0)

            # --- 5. Merge con gdf_division y calcular porcentaje usando el área real ---
            division_final = gdf_division.set_index('CVEGEO').join(pivot_area).fillna(0)
            for col in pivot_area.columns:
                division_final[col] = (division_final[col] / division_final.geometry.area)

            gdf_eda.drop(columns=['FACETAS', 'GRIETAS', 'GRAVAS', 'GUIJARROS', 'PIEDRAS', 'CONCRECIóN',
                'NóDULOS', 'MANCHAS', 'RAíCES', 'PRECIP', 'TEMP', 'LIM_SUP', 'LIM_INF'], inplace=True)
            gdf = gdf_eda.copy()

            # Crear coords como tupla
            gdf['coords'] = gdf.geometry.apply(lambda p: (p.x, p.y))

            # Separar columnas
            fixed_cols = [
                'ID_PERFIL','COORD_X','COORD_Y','ZONA','IDHOJA','FECHA',
                'CLAVE_WRB','GPO_SUELO','CALIF_PRIM','CALIF_SEC','F_RúDICA',
                'ALTITUD','GEOLOGíA','VEGETACIóN','PENDIENTE','RELIEVE','PEDREG','AFLORAM'
            ]

            excluded = set(fixed_cols + ['coords', gdf.geometry.name])
            layer_cols = [c for c in gdf.columns if c not in excluded]
            numeric_cols = [c for c in layer_cols if is_numeric_dtype(gdf[c])]
            cat_cols = [c for c in layer_cols if c not in numeric_cols]

            grp = gdf.groupby('coords', sort=False)

            # Función para obtener la primera moda
            def first_mode(series):
                s = series.dropna()
                if s.empty:
                    return np.nan
                m = s.mode()
                return m.iat[0] if len(m) > 0 else s.iat[0]

            # DataFrame final por coordenada
            df_final = pd.DataFrame({'coords': list(grp.groups.keys())})

            # Fijas
            df_fixed = grp[fixed_cols].first().reset_index(drop=True)
            df_final = pd.concat([df_final, df_fixed], axis=1)

            # Numéricas
            if numeric_cols:
                df_num = grp[numeric_cols].mean().reset_index(drop=True)
                df_final = pd.concat([df_final, df_num], axis=1)

            # Categóricas
            if cat_cols:
                df_cat = pd.DataFrame({c: grp[c].agg(first_mode) for c in cat_cols})
                df_final = pd.concat([df_final, df_cat.reset_index(drop=True)], axis=1)

            # Reconstruir geometría
            df_final['geometry'] = [Point(xy) for xy in df_final['coords']]
            gdf_final = gpd.GeoDataFrame(df_final, geometry='geometry', crs=gdf.crs)

            self.logger.info(f"Puntos originales: {len(gdf)}")
            self.logger.info(f"Puntos únicos por coordenada: {len(gdf_final)}")

            # -------------------- Parámetros --------------------
            ohe_cols = ["GPO_SUELO", "GEOLOGíA", "VEGETACIóN"]
            ohe_veg_suffix = "_eda"  # sufijo para vegetación
            fixed_cols = []  # si quieres conservar otras columnas como metadata
            # ----------------------------------------------------

            # 1) Intersectar puntos con polígonos
            gdf_inter = gpd.sjoin(gdf_final, division_final, how="inner", predicate="within")

            # 2) Preparar dataframe de resultados
            res_list = []

            # 2a) OHE counts por polígono
            ohe_frames = []
            for c in ohe_cols:
                col_suffix = ohe_veg_suffix if c == "VEGETACIóN" else ""
                ct = pd.crosstab(gdf_inter['CVEGEO'], gdf_inter[c]).astype(int)
                # renombrar columnas
                ct.columns = [f"{c}{col_suffix}__{str(col).replace(' ', '_').replace('/', '_')}" for col in ct.columns]
                ct = ct.reset_index()
                ohe_frames.append(ct)

            # merge OHE frames por CVEGEO
            if ohe_frames:
                ohe_df = ohe_frames[0]
                for df_ in ohe_frames[1:]:
                    ohe_df = ohe_df.merge(df_, on='CVEGEO', how='outer')
                ohe_df = ohe_df.fillna(0)
            else:
                ohe_df = pd.DataFrame({'CVEGEO': division_final.index})

            res_list.append(ohe_df)

            # 2b) Otras categóricas (moda)
            cat_cols = [c for c in gdf_final.columns if c not in ohe_cols + ['coords','geometry']]
            cat_cols = [c for c in cat_cols if not is_numeric_dtype(gdf_final[c])]
            if cat_cols:
                def first_mode(series):
                    s = series.dropna()
                    if s.empty:
                        return np.nan
                    m = s.mode()
                    return m.iat[0] if len(m) > 0 else s.iat[0]

                catmode_df = gdf_inter.groupby('CVEGEO')[cat_cols].agg(first_mode).reset_index()
                res_list.append(catmode_df)

            # 2c) Numéricas (media)
            num_cols = [c for c in gdf_final.columns if is_numeric_dtype(gdf_final[c])]
            if num_cols:
                num_df = gdf_inter.groupby('CVEGEO')[num_cols].mean().reset_index()
                res_list.append(num_df)

            # 3) Merge todos los agregados por CVEGEO
            df_agg = reduce(lambda left, right: pd.merge(left, right, on='CVEGEO', how='outer'), res_list)

            # 4) Unir con division_final para crear GeoDataFrame final
            gdf_result = division_final.merge(df_agg, left_index=True, right_on='CVEGEO', how='left')

            # Opcional: rellenar NaN de conteos con 0
            count_cols = [c for c in gdf_result.columns if '__' in c]
            gdf_result[count_cols] = gdf_result[count_cols].fillna(0).astype(int)

            # Resultado final
            self.logger.debug(f"Polígonos originales: {len(division_final)}")
            self.logger.debug(f"Polígonos con datos de puntos: {len(gdf_result)}")

            cols_useless = ['AREA', 'PERIMETER', 'COV_', 'COV_ID', 'ZONA','IDHOJA', 'ID_PERFIL']

            gdf_result.drop(columns=cols_useless, inplace = True)

            gdf_result = gdf_result.reset_index(drop=True)
            # -----------------------------------------------------------------------
            # Rellenar valores faltantes por vecinos más cercanos
            self.logger.info("Iniciando relleno de municipios con datos faltantes ...")
            # Parámetros ajustables
            k = self.params.get("k_neighbors", 5)                 # número de vecinos
            threshold_m = self.params.get("threshold_m", 20000)   # distancia máxima en metros (50 km)

            # Verifica CRS
            if gdf_result.crs.to_epsg() != 6372:
                self.logger.debug(f"Advertencia: tu CRS no es EPSG:6372 (actual: {gdf_result.crs}), considera convertirlo.")
            else:
                self.logger.debug("CRS correcto: EPSG:6372 (distancias en metros).")

            # Calcular centroides (para cada municipio)
            gdf_result['centroid'] = gdf_result.geometry.centroid
            coords = np.vstack([gdf_result['centroid'].x.values, gdf_result['centroid'].y.values]).T

            # Detectar columnas edafológicas (numéricas y categóricas)
            try:
                eda_num_cols = [c for c in num_cols if c in gdf_result.columns]
                eda_cat_cols = [c for c in cat_cols if c in gdf_result.columns]
            except NameError:
                eda_num_cols = [c for c in gdf_result.columns if is_numeric_dtype(gdf_result[c]) and '__' not in c]
                eda_cat_cols = [c for c in gdf_result.columns if (not is_numeric_dtype(gdf_result[c])) and c not in ['geometry','centroid','CVEGEO']]

            self.logger.info(f"Columnas numéricas: {eda_num_cols}")
            self.logger.info(f"Columnas categóricas: {eda_cat_cols}")

            # Construir modelo de vecinos más cercanos (una vez)
            def mode_or_nan(values):
                vals = [v for v in values if pd.notna(v)]
                if len(vals) == 0:
                    return np.nan
                return Counter(vals).most_common(1)[0][0]

            # --- Numéricas ---
            for col in eda_num_cols:
                is_nan = gdf_result[col].isna().values
                if not is_nan.any():
                    continue

                valid_idx = np.where(~is_nan)[0]
                missing_idx = np.where(is_nan)[0]
                if len(valid_idx) == 0:
                    self.logger.error(f" - No hay valores válidos para {col}")
                    continue

                coords_valid = coords[valid_idx]
                nbrs_valid = NearestNeighbors(n_neighbors=min(k, len(valid_idx))).fit(coords_valid)
                dists, inds = nbrs_valid.kneighbors(coords[missing_idx])

                for i_miss, (dist_row, ind_row) in enumerate(zip(dists, inds)):
                    neighbor_indices = valid_idx[ind_row]
                    if dist_row.min() <= threshold_m:
                        within = neighbor_indices[dist_row <= threshold_m]
                        vals = [v for v in gdf_result.iloc[within][col].values if pd.notna(v)]
                        if vals:
                            gdf_result.iloc[missing_idx[i_miss], gdf_result.columns.get_loc(col)] = np.mean(vals)

            # --- Categóricas ---
            for col in eda_cat_cols:
                is_nan = gdf_result[col].isna().values
                if not is_nan.any():
                    continue

                valid_idx = np.where(~is_nan)[0]
                missing_idx = np.where(is_nan)[0]
                if len(valid_idx) == 0:
                    self.logger.error(f" - No hay valores válidos para {col}")
                    continue

                coords_valid = coords[valid_idx]
                nbrs_valid = NearestNeighbors(n_neighbors=min(k, len(valid_idx))).fit(coords_valid)
                dists, inds = nbrs_valid.kneighbors(coords[missing_idx])

                for i_miss, (dist_row, ind_row) in enumerate(zip(dists, inds)):
                    neighbor_indices = valid_idx[ind_row]
                    if dist_row.min() <= threshold_m:
                        within = neighbor_indices[dist_row <= threshold_m]
                        vals = [v for v in gdf_result.iloc[within][col].values if pd.notna(v)]
                        m = mode_or_nan(vals)
                        if pd.notna(m):
                            gdf_result.iloc[missing_idx[i_miss], gdf_result.columns.get_loc(col)] = m

            # Limpieza
            gdf_result.drop(columns=['centroid', 'COORD_X', 'COORD_Y'], inplace=True)

            self.logger.info("Relleno de municipios completado.")

            # Reordenar columnas para que CVEGEO quede al inicio
            cols = gdf_result.columns.tolist()
            if 'CVEGEO' in cols:
                cols.insert(0, cols.pop(cols.index('CVEGEO')))
            gdf_result = gdf_result[cols]
            del cols
            gdf_result = self.downcast_numeric_gdf(gdf_result)
            output_path = os.path.join(self.preprocessed_data_dir, "suelo_merged.parquet")
            gdf_result.to_parquet(output_path)
        except Exception as e:
            self.logger.error(f"Hubo un error al unir los datos {e}")

def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        shp_division_path = os.path.join(root_dir,"data/Division_politica/mun22gw.shp")
        shp_eda_path = os.path.join(root_dir,"data/Edafologia_perfilsuelo/perfiles_suelos_shp/perfiles_serieii.shp")
        shp_suelo_vegetacion_path = os.path.join(root_dir,"data/Suelo_vegetacion/conjunto_de_datos/cdv_usuev250sVII_cnal.shp")
        output_path = os.path.join(root_dir,"data/processed/suelo")

        preprocessing = DataPreprocessingSoil(params_path=params_path,
                                            division_dir=shp_division_path,
                                            raw_eda_dir=shp_eda_path,
                                            raw_soil_dir=shp_suelo_vegetacion_path,
                                            preprocessed_data_dir=output_path)
        preprocessing.merge_division_soil()
    except Exception as e:
        preprocessing.logger.error(f"Failed to complete the data preprocessing pipeline {e}")

if __name__ == "__main__":
    main()