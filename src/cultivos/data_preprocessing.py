import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
from functools import reduce
from utils import BaseUtils, create_logger

class DataPreprocessingSeeds(BaseUtils):
    def __init__(self, params_path: str, sitios_path:str , output_path: str):
        logger = create_logger('data_preprocessing_seeds', 'logs/data_preprocessing_seeds_errors.log')
        super().__init__(logger=logger,params_path=params_path)
        self.sitios_path = sitios_path
        self.output_path = output_path
        self.params = self.load_params()["data_preprocessing_seeds"]
        self.final = None

    def evaluation_df(self) -> None:
        os.makedirs(self.output_path, exist_ok=True)  # exist_ok=True evita error si ya existe
        try:
                
            df = pd.read_csv(self.sitios_path,encoding='latin-1')
            df.drop(columns=['Idddr','Nomddr','Idcader','Nomcader','Idciclo','Idmodalidad','Idunidadmedida','Volumenproduccion','Preciomediorural'],inplace=True)

            df['Rendimiento'] = np.where(
                (df['Rendimiento'].isna()) & (df['Cosechada'] == 0),
                0,
                df['Rendimiento']
            )

            cultivos = df['Nomcultivo'].unique()
            indices_por_cultivo = {}

            for cultivo in cultivos:
                df_cultivo = df[df['Nomcultivo'] == cultivo].copy()

                # Agrupar por estado y municipio
                indice_df = df_cultivo.groupby(['Idestado','Idmunicipio']).agg(
                    Sembrada_total=('Sembrada','sum'),
                    Cosechada_total=('Cosechada','sum'),
                    Siniestrada_total=('Siniestrada','sum'),
                    Rendimiento_promedio=('Rendimiento','mean')
                ).reset_index()

                # Calcular índice productivo
                indice_df['Indice_productivo'] = np.where(
                    indice_df['Sembrada_total'] > 0,
                    indice_df['Rendimiento_promedio'] * (indice_df['Cosechada_total'] - indice_df['Siniestrada_total']) / indice_df['Sembrada_total'],
                    0
                )

                # Convertir negativos a 0
                indice_df['Indice_productivo'] = np.where(indice_df['Indice_productivo'] < 0, 0, indice_df['Indice_productivo'])

                # Normalizar dentro del cultivo
                mi, ma = indice_df['Indice_productivo'].min(), indice_df['Indice_productivo'].max()
                indice_df['Indice_norm'] = (indice_df['Indice_productivo'] - mi) / (ma - mi) if ma > mi else 0.0
                bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 5 rangos
                labels = ['Muy Malo', 'Malo', 'Regular', 'Bueno', 'Excelente']

                # Asignar etiqueta a cada municipio
                indice_df['Etiqueta'] = pd.cut(indice_df['Indice_norm'], bins=bins, labels=labels, include_lowest=True)
                # Ordenar por índice normalizado
                indice_df.sort_values(by='Indice_norm', ascending=False, inplace=True)

                # Guardar en diccionario
                indices_por_cultivo[cultivo] = self.downcast_numeric(indice_df)

            # Lista para guardar los DataFrames de cada cultivo ya con etiqueta
            df_list = []

            for cultivo in cultivos:
                df_cultivo = indices_por_cultivo[cultivo][['Idestado', 'Idmunicipio', 'Etiqueta']].copy()
                # Renombrar la columna de etiqueta al nombre del cultivo
                df_cultivo.rename(columns={'Etiqueta': cultivo}, inplace=True)
                # Convert the column to object type before filling NaN
                df_cultivo[cultivo] = df_cultivo[cultivo].astype('object')
                df_list.append(df_cultivo)

            # Unir todos los DataFrames por Idestado e Idmunicipio

            df_final = reduce(lambda left, right: pd.merge(left, right, on=['Idestado', 'Idmunicipio'], how='outer'), df_list)
            del df_list
            # Opcional: ordenar por estado y municipio
            df_final['CVEGEO'] = df_final['Idestado'].apply(lambda x: f"{x:02d}") + df_final['Idmunicipio'].apply(lambda x: f"{x:03d}")
            cols = ['CVEGEO'] + [c for c in df_final.columns if c != 'CVEGEO']
            df_final = df_final[cols]
            df_final.sort_values(by=['CVEGEO'], inplace=True)
            df_final.reset_index(drop=True, inplace=True)

            mapeo = {
                'Muy Malo': 1,
                'Malo': 2,
                'Regular': 3,
                'Bueno': 4,
                'Excelente': 5
            }

            df_final = df_final.map(lambda x: mapeo.get(x, x))
            df_final.fillna(0, inplace=True)

            # Me quedo con el df_final justo aqui, pero hay que hacer downcast
            df_final = self.downcast_numeric(df_final)
            output_evaluaciones_path = os.path.join(self.output_path, "evaluacion_cultivos.parquet")
            df_final.to_parquet(output_evaluaciones_path, index=False)
            self.df_final = df_final

        except Exception as e:
            raise self.logger.error(f"Hubo un error al crear el df de evaluación {e}")



    def key_municipalities(self) -> None:
        os.makedirs(self.output_path, exist_ok=True)  # exist_ok=True evita error si ya existe
        try:
            if self.df_final is None:
                raise ValueError("No existe el df de evaluación")
            bueno_excelente_per_municipio = []

            # Iterate over rows of df_final
            for index, row in self.df_final.iterrows():
                # Select columns excluding 'Idestado' and 'Idmunicipio'
                cultivo_ratings = row.drop(['Idestado', 'Idmunicipio'])

                # Count 'Bueno' and 'Excelente' in the row
                count = cultivo_ratings[(cultivo_ratings == 5)].count()
                #(cultivo_ratings == 'Bueno') |
                # Store the count along with state and municipality IDs
                bueno_excelente_per_municipio.append({
                    'CVEGEO': row['CVEGEO'],
                    'Idestado': row['Idestado'],
                    'Idmunicipio': row['Idmunicipio'],
                    'Count': count
                })

            # Create a DataFrame from the results
            df_bueno_excelente_per_municipio = pd.DataFrame(bueno_excelente_per_municipio)
            df_filtered = df_bueno_excelente_per_municipio[df_bueno_excelente_per_municipio['Count'] > self.params.get("lower_bound", 2)].copy()

            df_filtered.drop(columns=['Idestado', 'Idmunicipio', 'Count'], inplace=True)

            # Me quedo con la lista de CVEGEO que son interesantes, hacemos downcast la guardamos como df
            df_filtered = self.downcast_numeric(df_filtered)
            output_principales_path = os.path.join(self.output_path, "municipios_principales.parquet")
            df_filtered.to_parquet(output_principales_path, index=False)
        except Exception as e:
            self.logger.error(f"Hubo un fallo en la obtención de los municipios principales {e}")

def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        sitios_path = os.path.join(root_dir,"data/Sitios/Cierre_agricola_mun_2024.csv")
        output_path = os.path.join(root_dir,"data/processed/cultivos")

        preprocessing = DataPreprocessingSeeds(params_path=params_path,
                                               sitios_path=sitios_path,
                                               output_path=output_path)
        preprocessing.evaluation_df()
        preprocessing.key_municipalities()
    except Exception as e:
        raise preprocessing.logger.error(f"Failed to complete the data preprocessing pipeline {e}")

if __name__ == "__main__":
    main()