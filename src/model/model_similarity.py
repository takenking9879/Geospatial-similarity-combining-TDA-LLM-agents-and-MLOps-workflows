import pandas as pd
import numpy as np
import gower
import warnings
import gc
import math
import os
from gtda.homology import VietorisRipsPersistence
from gtda.metaestimators import CollectionTransformer
from gtda.time_series import TakensEmbedding
from gtda.diagrams import Scaler, PairwiseDistance
from gtda.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import time
from joblib import Parallel, delayed
from gtda.diagrams import Filtering
from utils import create_logger, BaseUtils
from homology import VietorisRipsPersistencePP

class TDA_similarity(BaseUtils):
    def __init__(self,root, serie_cols: list,
                 embedding_dimension: int = 100,
                 embedding_time_delay: int = 10,
                 stride: int = 2,
                 n_components: float = 3,
                 homology_dimensions: list = [0, 1],
                 metric: str = "wasserstein",
                 weights: list = None,
                 scaler=MinMaxScaler(),
                 epsilon: float = 0.05,
                 device: str = "cpu"):
        logger = create_logger('tda_similarity', 'logs/tda_similarity.log')
        super().__init__(logger=logger,params_path=None)
        self.weights = self.check_weights(weights)
        self.embedding_dimension = embedding_dimension
        self.embedding_time_delay = embedding_time_delay
        self.stride = stride
        self.metric = metric
        self.serie_cols = serie_cols
        self.n_components = n_components
        self.scaler = scaler
        self.homology_dimensions = homology_dimensions
        self.epsilon = epsilon
        self.device = device
        self.pipeline = self.build_topological_distance_pipeline()
        self.df_evaluacion = None
        self.confianza_path = None
        self.root = root
    def set_new_pipeline_params(self, **kwargs):
        """
        Actualiza los parámetros del pipeline de topología usando un diccionario de argumentos.
        kwargs puede incluir:
            - embedding_dimension
            - embedding_time_delay
        """
        if "embedding_dimension" in kwargs:
            self.embedding_dimension = kwargs["embedding_dimension"]
        if "embedding_time_delay" in kwargs:
            self.embedding_time_delay = kwargs["embedding_time_delay"]
        if "epsilon" in kwargs:
            self.epsilon = kwargs["epsilon"]

        self.pipeline = self.build_topological_distance_pipeline()

    def check_weights(self, weights: list):
        if weights is not None and not math.isclose(sum(weights), 1.0, rel_tol=1e-9):
            raise warnings.warn("La suma de los pesos no es igual a 1")
        return weights

    def build_topological_distance_pipeline(self):
        self.check_embedding_size()
        embedder = TakensEmbedding(
            time_delay=self.embedding_time_delay,
            dimension=self.embedding_dimension,
            stride=self.stride
        )
        batch_pca = CollectionTransformer(PCA(n_components=self.n_components, random_state=42), n_jobs=-1)
        persistence = VietorisRipsPersistence(homology_dimensions=self.homology_dimensions, n_jobs=-1)
        if self.device in ["gpu", "cuda", "GPU"]:
            persistence = VietorisRipsPersistencePP(homology_dimensions=self.homology_dimensions, n_jobs=-1)

        filtering =  Filtering(epsilon=self.epsilon) #7.5% aprox se quitan
        scaling = Scaler(n_jobs=-1)
        topo_pipeline = Pipeline([
            ("embedder", embedder),
            ("pca", batch_pca),
            ("persistence", persistence),
            ("scaling", scaling),
            ("filtering", filtering)
        ])
        return topo_pipeline

    def check_embedding_size(self) -> None:
        n_points = (len(self.serie_cols) - (self.embedding_dimension - 1) * self.embedding_time_delay) // self.stride
        n_points = max(int(n_points), 0)
        if n_points > 1500:
            warnings.warn(f"El embedding tendrá {n_points} puntos. Esto puede ser muy costoso.")
        else:
            self.logger.info(f"El embedding tendrá {n_points} puntos.")

    def cross_gower_distance(self, df1: pd.DataFrame, df2: pd.DataFrame,
                            categorical_cols: list, numeric_cols: list, weights: list = None) -> pd.DataFrame:
        try:
            ids1 = df1['CVEGEO'].astype(str)
            ids2 = df2['CVEGEO'].astype(str)
            X1 = df1[categorical_cols + numeric_cols].copy()
            X2 = df2[categorical_cols + numeric_cols].copy()
            
            # Cálculo de matriz de Gower
            D_cross = gower.gower_matrix(X1, X2, weight=weights)
            del X1, X2

            # Aplicar normalización usando tu función si D_cross no está entre 0 y 1
            if D_cross.min() < 0 or D_cross.max() > 1:
                self.logger.info("D_cross fuera de [0,1], normalizando usando self.normalization()")
                return self.normalization(pd.DataFrame(D_cross, index=ids1, columns=ids2))

            self.logger.info("Cálculo de matriz de Gower completado.")
            result = self.normalization(pd.DataFrame(D_cross, index=ids1, columns=ids2))
            return result 
        except Exception as e:                                      
            raise ValueError(f"Ocurrió un error calculando la matriz de Gower: {e}")

    def cross_pairwise_distance(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        # Chequeo si df1 y df2 tienen los mismos CVEGEO
        same_cvegeo = df1['CVEGEO'].equals(df2['CVEGEO'])
        self.logger.debug(f"Calculando distancias topológicas entre conjuntos con mismos CVEGEO: {same_cvegeo}")
        
        # Transformaciones
        if same_cvegeo:
            X1_transformed = self.pipeline.fit_transform(df1[self.serie_cols].to_numpy())
            X2_transformed = X1_transformed  # reutilizamos
        else:
            self.pipeline.fit(df1[self.serie_cols].to_numpy())
            X1_transformed = self.pipeline.transform(df1[self.serie_cols].to_numpy())
            X2_transformed = self.pipeline.transform(df2[self.serie_cols].to_numpy())
        
        # Información de tamaño de los embeddings
        self.logger.debug(f"X1_transformed: shape = {X1_transformed.shape}, tamaño en bytes = {X1_transformed.nbytes}")
        self.logger.debug(f"X2_transformed: shape = {X2_transformed.shape}, tamaño en bytes = {X2_transformed.nbytes}")

        self.logger.info("Transformaciones topológicas completadas.")
        start_time = time.time()  # ⏱ Empieza a contar

        # Calcular distancias topológicas
        pairwise = PairwiseDistance(metric=self.metric, n_jobs=-1)
        pairwise.fit(X1_transformed)
        D_cross = pairwise.transform(X2_transformed)
        D_cross = D_cross.T

        # Limpiar memoria
        del X1_transformed, X2_transformed
        self.logger.info("Cálculo de distancias topológicas completado.")
        gc.collect()
        end_time = time.time()  # ⏱ Termina de contar
        self.logger.info(f"Tiempo transcurrido: {end_time - start_time:.2f} segundos")
        
        # Crear DataFrame con índices y columnas como CVEGEO
        idx = df1['CVEGEO'].astype(str)
        cols = df2['CVEGEO'].astype(str)
        return pd.DataFrame(D_cross, index=idx, columns=cols)

    def row_mean_absolute_difference_matrix(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        idx = df1['CVEGEO'].astype(str)
        cols = df2['CVEGEO'].astype(str)
        means1 = df1[self.serie_cols].mean(axis=1).values
        means2 = df2[self.serie_cols].mean(axis=1).values
        diff_matrix = np.abs(means1[:, None] - means2[None, :])
        del means1, means2
        return pd.DataFrame(diff_matrix, index=idx, columns=cols)

    def hadamard_pairwise_lists(self, distances: list[pd.DataFrame], differences: list[pd.DataFrame]) -> list[pd.DataFrame]:
        if len(distances) != len(differences):
            raise ValueError("Las listas deben tener la misma longitud.")
        results = []
        for df_dist, df_diff in zip(distances, differences):
            if df_dist.shape != df_diff.shape:
                raise ValueError("Cada par de DataFrames debe coincidir en forma.")
            results.append(self.downcast_numeric(df_dist * df_diff))
        return results

    def save_tda_matrices(self, result, output_dir="cache/tda_matrix") -> None:
        """
        Guarda una lista de 3 DataFrames (tmax, tmin, precipitacion)
        en formato Parquet dentro de la carpeta especificada.
        Si la carpeta no existe, se crea automáticamente.
        """
        dir = os.path.join(self.root,output_dir)
        # Verificación básica
        if not isinstance(result, list) or len(result) != 3:
            raise ValueError("result debe ser una lista con 3 DataFrames: [tmax, tmin, precipitacion].")

        os.makedirs(output_dir, exist_ok=True)

        names = ["tmax", "tmin", "precipitacion"]
        for df, name in zip(result, names):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"El elemento '{name}' no es un DataFrame.")
            path = os.path.join(dir, f"{name}.parquet")
            df.to_parquet(path)
            print(f"✅ Guardado: {path} (shape={df.shape})")

    def load_tda_matrices(self, input_dir="cache/tda_matrix") -> list[pd.DataFrame]:
        """
        Carga los 3 DataFrames previamente guardados en la carpeta dada.
        Retorna una lista en el mismo orden: [tmax, tmin, precipitacion].
        """
        dir = os.path.join(self.root,input_dir)
        names = ["tmax", "tmin", "precipitacion"]
        dfs = []
        for name in names:
            path = os.path.join(dir, f"{name}.parquet")
            if not os.path.exists(path):
                raise FileNotFoundError(f"No se encontró el archivo {path}")
            df = pd.read_parquet(path)
            dfs.append(df)
            print(f"📂 Cargado: {path} (shape={df.shape})")
        return dfs

    def tda_matrix(self, df1s: list[pd.DataFrame], df2s: list[pd.DataFrame], new_params_list: list[dict] = None) -> list[pd.DataFrame]:
        if len(df1s) != len(df2s):
            raise ValueError("df1s y df2s deben tener la misma longitud")
        
        # Si no se pasan parámetros, usamos una lista de None
        if new_params_list is None:
            new_params_list = [None] * len(df1s)
        elif len(new_params_list) != len(df1s):
            raise ValueError("new_params_list debe tener la misma longitud que df1s y df2s")
        try:
            diffs = []
            topo_matrices = []

            for df1, df2, params in zip(df1s, df2s, new_params_list):
                if params is not None:
                    self.set_new_pipeline_params(**params)  # aplica diccionario
                topo_distance_matrix = self.cross_pairwise_distance(df1, df2)
                topo_distance_matrix = self.normalization(topo_distance_matrix)
                topo_distance_matrix = self.downcast_numeric(topo_distance_matrix)
                topo_matrices.append(topo_distance_matrix)
                del topo_distance_matrix

                diff = self.row_mean_absolute_difference_matrix(df1, df2)
                diff = self.normalization(diff)
                diff = self.downcast_numeric(diff)
                diffs.append(diff)
                del diff

            gc.collect()
            result = self.hadamard_pairwise_lists(topo_matrices, diffs)
            return result

        except Exception as e:
            self.logger.error(f"Ocurrió un error calculando la matriz topológica: {e}")
            raise

    def get_tda_gower_matrices(self, df1s_tda: list[pd.DataFrame], df2s_tda: list[pd.DataFrame],
                         df1_gower: pd.DataFrame, df2_gower: pd.DataFrame,
                         categorical_cols: list, numeric_cols: list, new_params_list: list[dict] = None,  # <-- lista de parámetros por par de df
                         gower_weights: list = None,  cache_path: str = "TDA_results/tda_matrix") -> None:
        try:
            if os.path.exists(cache_path):
                self.logger.info(f"📂 Cargando D_tda desde cache: {cache_path}")
                D_tda_series = self.load_tda_matrices(cache_path)
            else:
                self.logger.info("⚙️ Calculando D_tda...")
                D_tda_series = self.tda_matrix(df1s_tda, df2s_tda, new_params_list)
                self.save_tda_matrices(D_tda_series, cache_path)
                self.logger.info(f"✅ D_tda guardado en cache: {cache_path}")

            D_gower = self.cross_gower_distance(df1_gower, df2_gower, categorical_cols, numeric_cols, gower_weights)
            self.D_tda = D_tda_series
            self.D_gower = D_gower
        except Exception as e:
            self.logger.error(f"Ocurrió un error calculando los índices de similitud: {e}")
            raise

    # ------------------ Evaluación del modelo (nueva métrica basada en similitudes) ------------------
    def seed_similarity(self):
        """
        Calcula S_cultivos a partir de self.df_evaluacion y lo guarda en self.s_cultivos.
        También guarda el orden de los municipios en self.cvegeo_index.
        """
        cultivos = self.df_evaluacion.columns[3:]
        cultivos_df = self.df_evaluacion.set_index("CVEGEO")[cultivos]
        
        # Guardar el orden de CVEGEOS
        self.cvegeo_index = cultivos_df.index.to_list()

        A = cultivos_df.values.astype(np.float32)  # (n_rows, n_cultivos)
        n_rows, n_cultivos = A.shape

        shared_counts = np.zeros((n_rows, n_rows), dtype=np.int32)
        sum_abs_diff = np.zeros((n_rows, n_rows), dtype=np.float32)
        mask_A = (A != 0.0)

        for k in range(n_cultivos):
            a_k = A[:, k]
            ma = mask_A[:, k]
            if not ma.any():
                continue
            # Outer boolean mask
            shared_k = np.outer(ma.astype(np.uint8), ma.astype(np.uint8)).astype(bool)
            if not shared_k.any():
                continue
            diff_k = np.abs(a_k[:, None] - a_k[None, :])
            sum_abs_diff += np.where(shared_k, diff_k, 0.0)
            shared_counts += shared_k.astype(np.int32)

        S_cultivos = np.full((n_rows, n_rows), np.nan, dtype=np.float32)
        nonzero_mask = shared_counts > 0
        S_cultivos[nonzero_mask] = 1.0 - (sum_abs_diff[nonzero_mask] / (4.0 * shared_counts[nonzero_mask]))

        self.s_cultivos = S_cultivos

    def evaluar_modelo_allstats(self, D: pd.DataFrame):
        """
        Usa self.s_cultivos (precalulado) para evaluar y devolver un DataFrame
        con la 'confianza' en porcentaje para cada par (filas=index de D, cols=columns de D).
        Requiere que self.s_cultivos y self.cvegeo_index existan y correspondan a df_evaluacion.CVEGEO.
        """
        # ---- checks mínimos ----
        if not hasattr(self, "s_cultivos") or self.s_cultivos is None:
            raise RuntimeError("self.s_cultivos no está calculado. Llama a calcular_s_cultivos() primero.")
        if not hasattr(self, "cvegeo_index") or self.cvegeo_index is None:
            raise RuntimeError("self.cvegeo_index no está disponible. Debe guardarse cuando se calcule s_cultivos.")

        # mapa base desde cvegeo_index (orden en el que se calculó s_cultivos)
        cvegeo_to_row = {cve: idx for idx, cve in enumerate(self.cvegeo_index)}

        # verificar que todas las columnas de D están en el mapping
        missing_cols = [c for c in D.columns if c not in cvegeo_to_row]
        if missing_cols:
            raise KeyError(f"Algunas columnas de D no están en df_evaluacion.CVEGEO: {missing_cols[:10]}...")

        # construir índices de filas y columnas para extraer del self.s_cultivos
        D_index_list = D.index.to_list()
        row_map = [cvegeo_to_row[cve] for cve in D_index_list]  # filas de s_cultivos en el orden de D.index
        col_map = [cvegeo_to_row[c] for c in D.columns]         # columnas de s_cultivos en el orden de D.columns

        # sanity check shapes
        if self.s_cultivos.shape[0] < max(row_map) + 1 or self.s_cultivos.shape[1] < max(col_map) + 1:
            raise RuntimeError("self.s_cultivos tiene dimensiones incompatibles con los índices de D.")

        # Extraer la submatriz de S_cultivos que corresponde a D
        # (shape: n_rows x n_cols)
        S_for_D = self.s_cultivos[np.ix_(row_map, col_map)].astype(np.float32)

        # D como numpy
        D_vals = D.values.astype(np.float32)

        # calcular confianza (NaNs en S_for_D se propagan)
        S_diff = np.abs(D_vals - S_for_D)
        confianza = 1.0 - S_diff

        # construir DataFrame resultante con mismos índices/columnas que D
        confianza_df = pd.DataFrame(confianza, index=D.index, columns=D.columns)

        # estadísticas (por columna / municipio objetivo)
        confianza_stats = confianza_df.agg(['mean', 'std', 'min', 'max'], axis=0)
        confianza_stats_percent = confianza_stats * 100

        # promedio general (sobre columnas) en porcentaje
        general_stats = confianza_stats_percent.mean(axis=1)
        self.logger.info(f"\nRendimiento general del modelo (%):\n{general_stats}")

        # guardar en porcentaje si corresponde
        confianza_df_percent = confianza_df * 100
        confianza_df_percent = self.downcast_numeric(confianza_df_percent)
        if getattr(self, "confianza_path", None) is not None:
            try:
                confianza_df_percent.to_parquet(self.confianza_path, index=True)
            except Exception as e:
                self.logger.error(f"No se pudo guardar confianza_df_percent en parquet: {e}")

        return confianza_df_percent
    
    def evaluar_modelo(self, D: pd.DataFrame) -> float:
        """
        Versión liviana que reutiliza self.s_cultivos (precalculado).
        Devuelve un float: el promedio (ignorando NaN) de la matriz de 'confianza'
        calculada sobre las filas y columnas de D.
        """
        # ---- checks mínimos ----
        if not hasattr(self, "s_cultivos") or self.s_cultivos is None:
            raise RuntimeError("self.s_cultivos no está calculado. Llama a calcular_s_cultivos() primero.")
        if not hasattr(self, "cvegeo_index") or self.cvegeo_index is None:
            raise RuntimeError("self.cvegeo_index no está disponible. Debe guardarse cuando se calcule s_cultivos.")

        cvegeo_to_row = {cve: idx for idx, cve in enumerate(self.cvegeo_index)}
        missing_cols = [c for c in D.columns if c not in cvegeo_to_row]

        if missing_cols:
            raise KeyError(f"Algunas columnas de D no están en df_evaluacion.CVEGEO: {missing_cols[:10]}...")

        # mapear filas y columnas de D a índices en self.s_cultivos
        D_index_list = D.index.to_list()
        row_map = [cvegeo_to_row[cve] for cve in D_index_list]
        col_map = [cvegeo_to_row[c] for c in D.columns]

        # sanity check shapes
        if self.s_cultivos.shape[0] < max(row_map) + 1 or self.s_cultivos.shape[1] < max(col_map) + 1:
            raise RuntimeError("self.s_cultivos tiene dimensiones incompatibles con los índices de D.")

        # Extraer submatriz S para D
        S_for_D = self.s_cultivos[np.ix_(row_map, col_map)].astype(np.float32)

        # D como numpy
        D_vals = D.values.astype(np.float32)

        # calcular confianza
        S_diff = np.abs(D_vals - S_for_D)
        confianza = 1.0 - S_diff

        # promedio por columna (municipio objetivo), ignorando NaNs
        col_means = np.nanmean(confianza, axis=0)  # shape: (n_cols,)
        # promedio global de la métrica por municipio (float)
        return float(np.mean(col_means))

    def similarity_index_evaluation(self, df_evaluacion:pd.DataFrame = None, confianza_path: str = None, parameters_type= None,
                                    n_initial: int=1000,
                                    m_refine: int=10,
                                    refine_decay: float=0.90,
                                    seed: int = 42):
        self.confianza_path = confianza_path
        if df_evaluacion is None:
            raise ValueError("No se dieron datos para evaluar el modelo")
        if self.D_tda is None or self.D_gower is None:
            raise ValueError("No se han calculado las matrices de distancias")
        self.df_evaluacion = self.downcast_numeric(df_evaluacion)
        # Suponiendo que self.D_tda es una lista de DataFrames
        self.D_tda = [self.normalization(df) for df in self.D_tda]

        dfs = self.D_tda + [self.D_gower]
        D_similarity = self.weighted_similarity(dfs, parameters_type, n_initial, m_refine, refine_decay, seed)
        self.evaluar_modelo_allstats(D_similarity)
        del self.D_tda, self.D_gower
        self.logger.info("Cálculo de índices de similitud completado.")
        return self.downcast_numeric(D_similarity)

    def weighted_similarity(self, distance_matrices: list[pd.DataFrame], parameters_type=None,
                            n_initial: int=1000,
                            m_refine: int=50,
                            refine_decay:float=0.90,
                            seed: int = 42,
                            ) -> pd.DataFrame:

        n = len(distance_matrices)
        if n == 0:
            raise ValueError("La lista de matrices de distancia está vacía.")

        idx, cols = distance_matrices[0].index, distance_matrices[0].columns
        mats = []
        for D in distance_matrices:
            if D.shape != distance_matrices[0].shape:
                raise ValueError("Todas las matrices deben tener la misma forma.")
            mats.append(D.values.astype(float))

        # Si no se pide búsqueda, usar pesos existentes o uniformes
        if parameters_type != 'search':
            if self.weights is None:
                self.weights = [1 / n] * n
            elif len(self.weights) != n:
                raise ValueError("La cantidad de pesos no coincide con el número de matrices.")
            combined_distance = sum(w * D for w, D in zip(self.weights, distance_matrices))
            return pd.DataFrame(1 - combined_distance, index=idx, columns=cols)

        rng = np.random.default_rng(seed)

        self.seed_similarity()

        # ---- 1) Generar candidatos ----
        if n == 1:
            candidates = np.ones((1, 1))
        elif n == 2:
            ws = np.linspace(0.0, 1.0, n_initial)
            candidates = np.vstack([np.column_stack([w, 1 - w]) for w in ws])
        else:
            candidates = rng.dirichlet(np.ones(n), size=n_initial)

        best_score = -np.inf
        best_weights = None
        best_similarity = None

        def eval_weights(weights):
            combined = np.zeros_like(mats[0])
            for i in range(n):
                combined += weights[i] * mats[i]
            similarity = 1.0 - combined
            sim_df = pd.DataFrame(similarity, index=idx, columns=cols)
            #sim_df = self.normalization(sim_df)
            score = float(self.evaluar_modelo(sim_df))
            return score, weights

        # ---- 2) Evaluar candidatos ----
        results = Parallel(n_jobs=-2, prefer="processes")(
            delayed(eval_weights)(weights) for weights in candidates
        )

        for score, weights in results:
            if score > best_score:
                best_score = score
                best_weights = np.array(weights, dtype=float)

        # construir similarity_df para el mejor inicial
        combined = np.zeros_like(mats[0])
        for i in range(n):
            combined += best_weights[i] * mats[i]
        best_similarity = pd.DataFrame(1.0 - combined, index=idx, columns=cols)

        self.logger.info(f"✨ Mejor inicial -> Score: {best_score:.4f}, Pesos: {np.round(best_weights, 3)}")

        # ---- 3) Refinamiento local ----
        base_scale = 0.2
        neighbors_per_step = max(12, int(n_initial // 2))

        for step in range(m_refine):
            perturb_scale = (refine_decay ** step) * base_scale
            new_candidates = []
            for _ in range(neighbors_per_step):
                comp_scale = perturb_scale * (1.0 - best_weights)
                comp_scale = np.maximum(comp_scale, perturb_scale * 1e-3)
                perturb = rng.normal(0.0, comp_scale)
                new_w = best_weights + perturb
                new_w = np.clip(new_w, 0.0, None)
                s = new_w.sum()
                if s <= 0:
                    new_w = np.ones_like(new_w) / len(new_w)
                else:
                    new_w = new_w / s
                new_candidates.append(new_w)

            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(eval_weights)(weights) for weights in new_candidates
            )

            improved = False
            for score, weights in results:
                if score > best_score:
                    improved = True
                    best_score = score
                    best_weights = np.array(weights, dtype=float)
                    combined = np.zeros_like(mats[0])
                    for i in range(n):
                        combined += best_weights[i] * mats[i]
                    best_similarity = pd.DataFrame(1.0 - combined, index=idx, columns=cols)

            if improved:
                self.logger.info(f"  🔁 Step {step+1} mejora: {best_score:.4f} con {np.round(best_weights, 3)}")
            
        self.logger.info(f"🏁 Pesos finales búsqueda: {np.round(best_weights, 4)} con score {best_score:.4f}")

        # ---- 4) Evaluar tu candidato manual (aunque no sume 1) ----
        if self.weights is not None and len(self.weights) == n:
            candidate = np.array(self.weights, dtype=float)
            combined_manual = np.zeros_like(mats[0])
            for i in range(n):
                combined_manual += candidate[i] * mats[i]
            manual_similarity = pd.DataFrame(1.0 - combined_manual, index=idx, columns=cols)

            manual_score = float(self.evaluar_modelo(manual_similarity))
            self.logger.info(f"🧩 Candidato manual -> Score: {manual_score:.4f}, Pesos: {np.round(candidate, 3)}")

            # Si es mejor, reemplazamos el resultado final
            if manual_score > best_score:
                best_score = manual_score
                best_weights = candidate
                best_similarity = manual_similarity

        self.logger.info(f"🏆 Pesos finales elegidos: {np.round(best_weights, 4)} con score {best_score:.4f}")

        self.weights = best_weights.tolist()
        return best_similarity

    def normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        df_scaled = pd.DataFrame(self.scaler.fit_transform(df), index=df.index, columns=df.columns)
        return df_scaled
    
