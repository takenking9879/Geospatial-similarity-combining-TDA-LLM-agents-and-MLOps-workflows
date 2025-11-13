# prepare_geojson.py
import json
import geopandas as gpd
import pandas as pd
import os
from utils import create_logger, BaseUtils
from collections import defaultdict

class PreprocessingAppData(BaseUtils):
    def __init__(self, shape_path:str, sim_path:str, soil_path:str, evaluation_path:str , output_dir:str):
        logger = create_logger('preprocess_app_data', 'logs/preprocess_app_data_errors.log')
        super().__init__(logger=logger,params_path=None)
        self.shape_path = shape_path
        self.sim_path = sim_path
        self.output_dir = output_dir
        #Data needed for Agents system 1
        self.soil_path = soil_path
        self.evaluation_path = evaluation_path

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            self.logger.info("🔧 Cargando shapefile y limpiando columnas...")
            # Leer shapefile
            gdf_div = gpd.read_file(self.shape_path, encoding="latin-1")

            # --- Limpieza de nombres de columnas ---
            if "CVEGEO" not in gdf_div.columns:
                candidates = [c for c in gdf_div.columns if c.upper().startswith("CVE")]
                if candidates:
                    gdf_div = gdf_div.rename(columns={candidates[0]: "CVEGEO"})
            if "NOMGEO" not in gdf_div.columns:
                for cand in ["NOM_MUN", "NOM_MUNICIPIO", "NOMBRE", "NOM"]:
                    if cand in gdf_div.columns:
                        gdf_div = gdf_div.rename(columns={cand: "NOMGEO"})
                        break
            if "ENTIDAD" not in gdf_div.columns and "NOM_ENT" in gdf_div.columns:
                gdf_div = gdf_div.rename(columns={"NOM_ENT": "ENTIDAD"})
            if "ENTIDAD" not in gdf_div.columns:
                for cand in ["STATE", "STATE_NAME", "estado", "ESTADO", "ENT"]:
                    if cand in gdf_div.columns:
                        gdf_div = gdf_div.rename(columns={cand: "ENTIDAD"})
                        break

            # --- Tipos ---
            gdf_div["CVEGEO"] = gdf_div["CVEGEO"].astype(str)
            gdf_div["ENTIDAD"] = gdf_div.get("ENTIDAD", "Unknown").astype(str)

            print("🗺️ Simplificando geometrías...")
            gdf_simp = gdf_div.to_crs(epsg=3857)
            gdf_simp["geometry"] = gdf_simp["geometry"].simplify(tolerance=100)
            gdf_simp = gdf_simp.to_crs(epsg=4326)

            if "NOM_ENT" not in gdf_simp.columns:
                gdf_simp["NOM_ENT"] = gdf_simp["ENTIDAD"]

            # --- Crear GeoJSON liviano ---
            geojson_obj = json.loads(gdf_simp.to_json())
            for feat in geojson_obj["features"]:
                props = feat["properties"]
                feat["properties"] = {
                    "CVEGEO": str(props.get("CVEGEO")),
                    "NOM_ENT": str(props.get("NOM_ENT", "")),
                    "NOMGEO": props.get("NOMGEO", ""),
                    "ENTIDAD": props.get("ENTIDAD", "")
                }

            # --- Guardar GeoJSON simplificado ---
            geojson_path = os.path.join(self.output_dir, "geojson_light.json")
            with open(geojson_path, "w", encoding="utf-8") as f:
                json.dump(geojson_obj, f)
            self.logger.info(f"✅ GeoJSON simplificado guardado en {geojson_path}")

            # --- Crear mapeo de estados a municipios ---
            self.logger.info("📍 Generando mapeo de estados...")
            df_similarity = pd.read_parquet(self.sim_path)
            df_similarity.columns = df_similarity.columns.astype(str)
            states = sorted(gdf_simp["ENTIDAD"].unique())
            state_to_muns = {}

            for st in states:
                muns = gdf_simp[gdf_simp["ENTIDAD"] == st][["CVEGEO", "NOM_ENT", "NOMGEO"]]
                muns = muns[muns["CVEGEO"].isin(df_similarity.columns)]
                state_to_muns[st] = muns.to_dict(orient="records")

            state_path = os.path.join(self.output_dir, "state_to_muns.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state_to_muns, f)

            self.logger.info(f"✅ Diccionario estado→municipios guardado en {state_path}")
            self.logger.info("🎉 Preparación completa.")
        except Exception as e:
            self.logger.error(f"Hubo un error al preprocesar los datos {e}")
            raise
    def system_agentic_data_analysis(self):
        df = pd.read_parquet(self.evaluation_path)
        suelo = gpd.read_parquet(self.soil_path)
        df_similarity = pd.read_parquet(self.sim_path)
        #Only keeping essential columns
        cols_a_conservar = suelo.columns[[0, 3, 4]]  # 1, 4, 5 (índices 0-based)
        cols_a_conservar = cols_a_conservar.append(suelo.columns[119:]) 
        suelo = suelo[cols_a_conservar].dropna()

        #Only keeping CVEGEO that have soil data and can be evaluated
        comunes = df['CVEGEO'].isin(suelo['CVEGEO'])
        df = df[comunes]
        suelo = suelo[suelo['CVEGEO'].isin(df['CVEGEO'])]
        self.logger.debug(f"CVEGEO restantes: {suelo.shape[0]}")
        self.logger.debug(f"CVEGEO restantes: {df.shape[0]}")

        # Diccionario para agrupar por municipio
        fixed_columns = ["CVEGEO", "Idestado", "Idmunicipio"]
        municipios_json = []
        for _, row in df.iterrows():
            municipio_data = {col: row[col] for col in fixed_columns}
            
            # Solo agregar cultivos con valor > 0
            cultivos = {col: row[col] for col in df.columns if col not in fixed_columns and row[col] > 0}
            
            municipio_data["cultivos"] = cultivos
            municipios_json.append(municipio_data)
        del municipio_data
        # Convertimos a JSON y guardamos
        municipios_json_path = os.path.join(self.output_dir, 'municipios.json')
        with open(municipios_json_path, "w", encoding="utf-8") as f:
            json.dump(municipios_json, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Se ha guardado el json de municipios")

        # Diccionario para agrupar por cultivo
        cultivos_dict = defaultdict(list)
        for _, row in df.iterrows():
            for col in df.columns:
                if col not in fixed_columns and row[col] > 0:
                    cultivos_dict[col].append({
                        "CVEGEO": row["CVEGEO"],
                        "Idestado": row["Idestado"],
                        "Idmunicipio": row["Idmunicipio"],
                        "valor": row[col]
                    })

        # Convertir a JSON y guardamos
        cultivos_json_path = os.path.join(self.output_dir, 'cultivos.json')
        with open(cultivos_json_path, "w", encoding="utf-8") as f:
            json.dump(cultivos_dict, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Se ha guardado el json de cultivos")

        columns_for_agent = [
            "CVEGEO", "NOMGEO", "NOM_ENT",
            "ALTITUD",
            "PH", "CE", "CO", "CIC", "SB", "SNA",
            "K", "CA", "NA", "MG", "CACO3", "CASO4",
            "DREN_EXT", "DREN_INT",
            "R", "L", "A"
        ]
        suelo = suelo[columns_for_agent]
        if 'geometry' in suelo.columns:
            suelo = pd.DataFrame(suelo.drop(columns='geometry'))
        
        new_soil_path = os.path.join(self.output_dir, 'soil.parquet')
        suelo.to_parquet(new_soil_path)
        self.logger.info(f"Se ha guardado el df de suelo recortado")

        lista_cultivos = list(df.columns)
        lista_cultivos = lista_cultivos[3:]
        lista_municipios = list(df_similarity.columns)

        df_filtrado = suelo[suelo["CVEGEO"].isin(lista_municipios)]
        # Crear diccionario en el formato deseado
        dict_cvegeo = {
            row["CVEGEO"]: f"{row['NOMGEO']}, {row['NOM_ENT']}"
            for _, row in df_filtrado.iterrows()
        }
        
        # Guardar dict_cvegeo
        cvegeo_json_path = os.path.join(self.output_dir, 'dict_cvegeo.json')
        with open(cvegeo_json_path, "w", encoding="utf-8") as f:
            json.dump(dict_cvegeo, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Se ha guardado el dict de los CVEGEO")

        # Guardar lista de cultivos
        lista_cultivos_json_path = os.path.join(self.output_dir, 'lista_cultivos.json')
        with open(lista_cultivos_json_path, "w", encoding="utf-8") as f:
            json.dump(lista_cultivos, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Se ha guardado la lista de cultivos")

def main():
    # Paths
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

        shape_path =  os.path.join(root_dir,'data/Division_politica/mun22gw.shp')
        sim_path = os.path.join(root_dir, 'flask_app/results/similarity_matrix.parquet')
        soil_path =  os.path.join(root_dir,'data/processed/suelo/suelo_merged.parquet')
        evaluation_path = os.path.join(root_dir,'data/processed/cultivos/evaluacion_cultivos.parquet')
        output_dir = os.path.join(root_dir, 'flask_app/results')
        preprocessing = PreprocessingAppData(shape_path=shape_path, sim_path=sim_path,
                                             soil_path=soil_path, evaluation_path=evaluation_path, output_dir=output_dir)
        preprocessing.run()
        preprocessing.system_agentic_data_analysis()
    except Exception as e:
        preprocessing.logger.error(f"Failed to complete the preprocessing pipeline {e}")

if __name__ == "__main__":
    main()