import pandas as pd
import numpy as np
from gtda.time_series import takens_embedding_optimal_parameters
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
from joblib import Parallel, delayed
from model.model_similarity import TDA_similarity
from utils import BaseUtils, create_logger
import os

class ModelBuilding(BaseUtils):
    def __init__(self, params_path: str, clima_path:str,soil_path:str , evaluation_path:str, principales_path:str, root:str, cache:str, confianza_path:str):
        logger = create_logger('model_building', 'logs/model_building_errors.log')
        super().__init__(logger=logger,params_path=params_path)
        self.clima_path = clima_path
        self.soil_path = soil_path
        self.evaluation_path = evaluation_path
        self.principales_path = principales_path
        self.params = self.load_params()["model_building"]
        self.root = root
        self.cache = cache
        self.confianza_path = confianza_path

    def find_best_embedding_mean(self,
        df: pd.DataFrame,
        serie_cols: list,
        max_time_delay: int,
        max_dimension: int,
        stride: int = 1,
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        Encuentra los parámetros Takens óptimos para cada serie, luego calcula el promedio global.
        
        Retorna un DataFrame con los promedios y desviaciones estándar de time_delay y dimension.
        """
        X = df[serie_cols].to_numpy()

        def process_column(col_idx: int):
            ts = X[:, col_idx].reshape(-1, 1)
            try:
                time_delay, dimension = takens_embedding_optimal_parameters(
                    ts,
                    max_time_delay=max_time_delay,
                    max_dimension=max_dimension,
                    stride=stride,
                    n_jobs=1,          # evitar paralelismo anidado
                    validate=True
                )
            except Exception:
                time_delay, dimension = np.nan, np.nan
            return time_delay, dimension

        # Progreso paralelo
        with tqdm_joblib(tqdm(desc="Buscando parámetros Takens", total=len(serie_cols))):
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_column)(col_idx) for col_idx in range(X.shape[1])
            )

        # Convertir a DataFrame
        results_df = pd.DataFrame(results, columns=["time_delay", "dimension"])
        valid = results_df.dropna()

        # Calcular promedio y desviación
        summary = pd.DataFrame({
            "time_delay_mean": [valid["time_delay"].mean()],
            "time_delay_std": [valid["time_delay"].std()],
            "dimension_mean": [valid["dimension"].mean()],
            "dimension_std": [valid["dimension"].std()],
            "n_valid": [len(valid)],
            "n_total": [len(results_df)]
        })
        return summary
    
    def model_data(self):
        try:
            # ------------------ Cargar datos ------------------
            df_series = pd.read_parquet(self.clima_path)
            df_info = pd.read_parquet(self.soil_path)
            df_evaluacion = pd.read_parquet(self.evaluation_path)
            df_principales = pd.read_parquet(self.principales_path)

            # ------------------ Preparación de df_info ------------------
            if 'geometry' in df_info.columns:
                df_info.drop(columns=['geometry'], inplace=True)

            # Mantener solo CVEGEO + variables
            keep_info_cols = [c for c in df_info.columns if c not in ['FECHA','CVE_ENT','CVE_MUN','NOMGEO','NOM_ENT']]
            df_info = df_info[keep_info_cols]
            del df_series['CVE_ENT'], df_series['CVE_MUN'], df_series['NOMGEO'], df_series['NOM_ENT']

            # Separar categóricas y numéricas
            categorical_cols = df_info.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = df_info.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
            categorical_cols = [c for c in categorical_cols if c != 'CVEGEO']  # no incluir identificador

            # ------------------ Preparación de df_series ------------------
            df_info.sort_values(by=['CVEGEO'], inplace=True)
            df_series.sort_values(by=['CVEGEO'], inplace=True)
            df_evaluacion.sort_values(by=['CVEGEO'], inplace=True)
            df_principales.sort_values(by=['CVEGEO'], inplace=True)
            df_info.reset_index(drop=True, inplace=True)
            df_series.reset_index(drop=True, inplace=True)
            df_evaluacion.reset_index(drop=True, inplace=True)
            df_principales.reset_index(drop=True, inplace=True)

            serie_cols = [col for col in df_series.columns if col not in ['CVEGEO', 'Variable']]

            # Identificar CVEGEO con NULLs y eliminar esas filas
            # Para df_series
            mask_null_series = df_series.isna().any(axis=1)
            cvegeo_nulls_series = df_series.loc[mask_null_series, 'CVEGEO'].tolist()

            # Para df_info
            mask_null_info = df_info.isna().any(axis=1)
            cvegeo_nulls_info = df_info.loc[mask_null_info, 'CVEGEO'].tolist()
            # 1️⃣ Combinar las listas y obtener valores únicos
            cvegeo_a_eliminar = list(set(cvegeo_nulls_series + cvegeo_nulls_info))
            self.logger.info(f"CVEGEO a eliminar: {len(set(cvegeo_a_eliminar))}")

            # 2️⃣ Filtrar ambos DataFrames para eliminar esas filas
            df_series = df_series[~df_series['CVEGEO'].isin(cvegeo_a_eliminar)].reset_index(drop=True)
            df_info   = df_info[~df_info['CVEGEO'].isin(cvegeo_a_eliminar)].reset_index(drop=True)
            df_info = df_info[df_info['CVEGEO'].isin(df_evaluacion['CVEGEO'])].reset_index(drop=True)
            df_series = df_series[df_series['CVEGEO'].isin(df_evaluacion['CVEGEO'])].reset_index(drop=True)
            df_evaluacion = df_evaluacion[df_evaluacion['CVEGEO'].isin(df_info['CVEGEO'])].reset_index(drop=True)
            df_principales = df_principales[df_principales['CVEGEO'].isin(df_info['CVEGEO'])].reset_index(drop=True)

            # ✅ Comprobar
            self.logger.info(f"Filas restantes en df_series: {len(df_series)//3}")
            self.logger.info(f"Filas restantes en df_info: {len(df_info)}")
            self.logger.info(f"Filas restantes en df_principales: {len(df_principales)}")
            self.logger.info(f"Filas restantes en df_evaluacion: {len(df_evaluacion)}")

            # Separar series por variable
            df_tmax = df_series[df_series['Variable'] == 'TMAX'].reset_index(drop=True)
            df_tmin = df_series[df_series['Variable'] == 'TMIN'].reset_index(drop=True)
            df_prep = df_series[df_series['Variable'] == 'PRECIP'].reset_index(drop=True)

            del df_series
            self.logger.debug(f"Es {(df_info['CVEGEO'].values == df_tmax['CVEGEO'].values).all()} que CVEGEO de df_info y df_tmax coinciden.")
            self.logger.debug(f"Es {(df_info['CVEGEO'].values == df_tmin['CVEGEO'].values).all()} que CVEGEO de df_info y df_tmin coinciden.")
            self.logger.debug(f"Es {(df_info['CVEGEO'].values == df_prep['CVEGEO'].values).all()} que CVEGEO de df_info y df_prep coinciden.")
            self.logger.debug(f"Es {(df_info['CVEGEO'].values == df_evaluacion['CVEGEO'].values).all()} que CVEGEO de df_info y df_evaluacion coinciden.")
            # ------------------ Preparación de listas para TDA ------------------
            df1s = [df_tmax, df_tmin, df_prep]  # origen

            #Calcular los embbedings optimos para cada tipo de serie
            #for i, df in enumerate(df1s):  # agrego enumerate para identificar el df
            #    result_optimal = find_best_embedding_mean(df, serie_cols, 50, 50, 3)
            #    print(f"\nEl mejor embedding para el DataFrame {i} es:\n{result_optimal.to_string(index=False)}")

            # ------------------ Filtrar para municipios principales ------------------
            df_tmax = df_tmax[df_tmax['CVEGEO'].isin(df_principales['CVEGEO'])].reset_index(drop=True)
            df_tmin = df_tmin[df_tmin['CVEGEO'].isin(df_principales['CVEGEO'])].reset_index(drop=True)
            df_prep = df_prep[df_prep['CVEGEO'].isin(df_principales['CVEGEO'])].reset_index(drop=True)

            df_info_principales = df_info[df_info['CVEGEO'].isin(df_principales['CVEGEO'])].reset_index(drop=True)
            
            df2s = [df_tmax, df_tmin, df_prep]  # objetivo (puede ser subset de municipios principales)
            del df_tmax, df_tmin, df_prep

            # ------------------ Crear clase TDA ------------------
            self.general_params = self.params["general_params"]
            dimension = self.general_params.get("dim", 8) #General con todas 8, 64, 3
            time_delay = self.general_params.get("time", 32)
            stride = self.general_params.get("stride", 1)
            similarity = TDA_similarity(root = self.root,
                serie_cols=serie_cols,
                embedding_dimension=dimension,
                embedding_time_delay=time_delay,
                stride=stride,
                metric=self.params.get("metric", "wasserstein"),
                weights=self.params.get("candidate", [0.1, 0.15, 0.4, 0.35]),
                device = self.params.get("device", "gpu") #[0.47, 0.53]
                ) #metric = 'bottleneck'

            # ------------------ Calcular similitud ------------------
            #Particular Tmax 8, 23, 0.08, Tmin 8, 23, 0.085. Precipitación 12, 21, 0.05
            new_params_list = self.params.get("params_list", [
            {"embedding_dimension": 7, "embedding_time_delay": 28, "epsilon": 0.07},   # Tmax
            {"embedding_dimension": 7, "embedding_time_delay": 29, "epsilon": 0.075},  # Tmin
            {"embedding_dimension": 14, "embedding_time_delay": 33, "epsilon": 0.04}   # Precip
            ])

            similarity.get_tda_gower_matrices(
                    df1s_tda=df1s,
                    df2s_tda=df2s,
                    df1_gower=df_info,
                    df2_gower=df_info_principales,
                    categorical_cols=categorical_cols,
                    numeric_cols=numeric_cols,
                    cache_path=self.cache,
                    new_params_list=new_params_list
                )
            optimize = self.params["optimize"]
            D = similarity.similarity_index_evaluation(df_evaluacion=df_evaluacion,confianza_path=self.confianza_path,
                                                       parameters_type=optimize.get('parameters_type','search'), n_initial=optimize.get("n_initial", 1000),
                                                       m_refine=optimize.get("m_refine", 50), refine_decay=optimize.get("refine_decay", 0.9),
                                                       seed = optimize.get("seed", 42))
            return D
        except Exception as e:
            self.logger.error(f"Hubo un error al calcular la similitud")
            raise

def main():
    try:
        # ------------------ Rutas ------------------
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        results_path = os.path.join(root_dir, 'flask_app/results')
        os.makedirs(results_path, exist_ok=True)
        
        params_path = os.path.join(root_dir, 'params.yaml')
        cache = os.path.join(root_dir, 'cache/tda_matrix')
        similarity_matrix_path = os.path.join(root_dir, 'flask_app/results/similarity_matrix.parquet')
        confianza_path = os.path.join(root_dir, 'flask_app/results/confianza_matrix.parquet')
        clima_path = os.path.join(root_dir, 'data/processed/clima/climatologia_merged.parquet')
        soil_path = os.path.join(root_dir, 'data/processed/suelo/suelo_merged.parquet')

        evaluation_path = os.path.join(root_dir, 'data/processed/cultivos/evaluacion_cultivos.parquet')
        principales_path = os.path.join(root_dir, 'data/processed/cultivos/municipios_principales.parquet')

        building = ModelBuilding(params_path=params_path, clima_path=clima_path,
                                soil_path=soil_path,evaluation_path=evaluation_path,
                                principales_path=principales_path, root=root_dir,
                                cache=cache, confianza_path=confianza_path)
        D = building.model_data()
        D.to_parquet(similarity_matrix_path, index=True)
        building.logger.info("✅ Matriz de similitud evaluada y guardada.")
        
        # --- ruta del output auxiliar ---
        aux_output_dir = os.path.join(root_dir, "output_aux")
        aux_output_path = os.path.join(aux_output_dir, 'aux_model_building.txt')
        # crear carpeta si no existe
        os.makedirs(aux_output_dir, exist_ok=True)

        # escribir algo simple en el archivo
        with open(aux_output_path, "w") as f:
            f.write("Proceso de preprocesamiento completado.\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        
    except Exception as e:
        building.logger.error(f"Failed to complete the model building pipeline {e}")
if __name__ == "__main__":
    main()
    #Importante asegurarse que si quieres recalcular la matrix de similitud, es necesario borrar el archivo manualmente