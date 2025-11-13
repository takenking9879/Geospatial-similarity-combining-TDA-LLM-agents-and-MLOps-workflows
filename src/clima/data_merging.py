import os
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import io
from utils import BaseUtils, create_logger

class DataMergingClimate(BaseUtils):
    def __init__(self, params_path: str, data_dir:str , output_dir: str):
        logger = create_logger('data_merging_climate', 'logs/data_merging_climate_errors.log')
        super().__init__(logger=logger,params_path=params_path)
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.params = self.load_params()["data_merging_climate"]

    def clean_float(self, s):
        try:
            return float(str(s).replace("Â","").replace("\xa0","").strip())
        except:
            return None
    
    def procesar_archivo_wide(self, file_path) -> pd.DataFrame:
        try:
            
            start_date = datetime(*self.params.get("start_date", [2013, 1, 1]))
            end_date   = datetime(*self.params.get("end_date", [2024, 12, 31]))
            date_range = pd.date_range(start_date, end_date)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Extraer metadatos
            meta = {"Estacion": None, "Nombre": None, "Estado": None, "Municipio": None,
                    "Latitud": None, "Longitud": None, "Altitud": None}
            for line in lines:
                line_clean = line.strip().replace("\xa0","").replace("Â","")
                if line_clean.startswith("ESTACIÓN"):
                    meta["Estacion"] = line_clean.split(":")[1].strip()
                elif line_clean.startswith("NOMBRE"):
                    meta["Nombre"] = line_clean.split(":")[1].strip()
                elif line_clean.startswith("ESTADO"):
                    meta["Estado"] = line_clean.split(":")[1].strip()
                elif line_clean.startswith("MUNICIPIO"):
                    meta["Municipio"] = line_clean.split(":")[1].strip()
                elif line_clean.startswith("LATITUD"):
                    meta["Latitud"] = self.clean_float(line_clean.split(":")[1].split("°")[0])
                elif line_clean.startswith("LONGITUD"):
                    meta["Longitud"] = self.clean_float(line_clean.split(":")[1].split("°")[0])
                elif line_clean.startswith("ALTITUD"):
                    meta["Altitud"] = self.clean_float(line_clean.split(":")[1].split("msnm")[0])

            # Detectar inicio de tabla
            start_idx = next((i for i, l in enumerate(lines) if l.strip().startswith("FECHA")), None)
            if start_idx is None:
                return None

            tabla_str = "".join(lines[start_idx:])
            df = pd.read_csv(io.StringIO(tabla_str), sep='\s+', na_values=["NULO"])
            df["FECHA"] = pd.to_datetime(df["FECHA"], format="%Y-%m-%d", errors="coerce")
            df = df.dropna(subset=["FECHA"])
            df = df[(df["FECHA"] >= start_date) & (df["FECHA"] <= end_date)]

            # Rellenar fechas faltantes
            df_full = pd.DataFrame({"FECHA": date_range})
            df = pd.merge(df_full, df, on="FECHA", how="left")

            # Crear filas por variable
            variables = ["PRECIP", "EVAP", "TMAX", "TMIN"]
            rows = []
            for var in variables:
                row = {
                    "Longitud": meta["Longitud"],
                    "Latitud": meta["Latitud"],
                    "Altitud": meta["Altitud"],
                    "Estacion": meta["Estacion"],
                    "Estado": meta["Estado"],
                    "Municipio": meta["Municipio"],
                    "Variable": var
                }
                for date in date_range:
                    date_str = date.strftime("%Y-%m-%d")
                    if var in df.columns:
                        value = df.loc[df["FECHA"] == date, var].values
                        row[date_str] = value[0] if len(value) > 0 else None
                    else:
                        row[date_str] = None
                rows.append(row)

            return pd.DataFrame(rows)
        except Exception as e:
            raise self.logger.error(f"Hubo un error al procesar los archivos {e}")
    
    def create_csv(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            n_cpus = os.cpu_count()
            max_workers = max(1, n_cpus - 1)
            output_path = os.path.join(self.output_dir, "climatologia_todos_los_archivos.csv")

            txt_files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]
            self.logger.info(f"Se encontraron {len(txt_files)} archivos .txt.")

            first_file = True
            chunk_size = 500  # cantidad de archivos a procesar antes de escribir
            total_files = len(txt_files)

            for i in range(0, total_files, chunk_size):
                chunk_files = txt_files[i:i + chunk_size]
                dfs = []

                # Procesamos el chunk en paralelo
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    for df_result in tqdm(
                            executor.map(self.procesar_archivo_wide,
                                        [os.path.join(self.data_dir, f) for f in chunk_files]),
                            total=len(chunk_files),
                            desc=f"Procesando archivos {i+1}-{i+len(chunk_files)}"):
                        if df_result is not None and not df_result.empty:
                            dfs.append(df_result)

                # Concatenamos los DataFrames del chunk y escribimos al CSV
                if dfs:
                    chunk_df = pd.concat(dfs, ignore_index=True)
                    chunk_df.to_csv(output_path, mode='w' if first_file else 'a', 
                                    index=False, header=first_file, encoding='utf-8')
                    first_file = False

            self.logger.info(f"✅ Todos los archivos combinados se guardaron en {output_path}")

        except Exception as e:
            self.logger.error(f"Hubo un error al crear el CSV: {e}")
            raise
def main():
    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params_path = os.path.join(root_dir, 'params.yaml')
        # Rutas de datos y output
        data_dir = os.path.join(root_dir, 'data/datos_climatologicos')
        output_dir = os.path.join(root_dir, 'data/processed/clima')

        merging = DataMergingClimate(params_path=params_path, data_dir=data_dir, output_dir=output_dir)
        merging.create_csv()
    except Exception as e:
        raise merging.logger.error(f"Failed to complete the data merging pipeline {e}")

if __name__ == "__main__":
    main()