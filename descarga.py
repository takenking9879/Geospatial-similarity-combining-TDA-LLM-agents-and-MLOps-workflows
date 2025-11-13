import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm
import geopandas as gpd

# Ahora localiza el archivo .kml
kml_path = "doc.kml"  # Generalmente se llama doc.kml

# Leer el KML con GeoPandas
gdf = gpd.read_file(kml_path, driver='KML', layer="Estaciones operativas")

# Crear carpeta
os.makedirs("data/datos_climatologicos", exist_ok=True)

estados_abrev = ['ags','bc','bcs','camp','coah','col','chps','chih','cdmx','dgo','gto',
                 'gro','hgo','jal','mex','mich','mor','nay','nl','oax','pue','qro','qroo',
                 'slp','sin','son','tab','tlax','ver','yuc','zac']

estaciones = [str(e).zfill(5) for e in gdf['Name'].to_list()]

# Función para descargar un archivo
def descargar(url, filename):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(filename, "wb") as f:
                f.write(r.content)
            return True
    except:
        return False
    return False

# Crear lista de trabajos
trabajos = []
for estado in estados_abrev:
    for est_id in estaciones:
        url = f"https://smn.conagua.gob.mx/tools/RESOURCES/Normales_Climatologicas/Diarios/{estado}/dia{est_id}.txt"
        filename = f"data/datos_climatologicos/{estado}_dia{est_id}.txt"
        trabajos.append((url, filename))

# Paralelizar
with ThreadPoolExecutor(max_workers=10) as executor:  # Ajusta max_workers según tu internet
    futures = {executor.submit(descargar, url, filename): (url, filename) for url, filename in trabajos}
    for f in tqdm(as_completed(futures), total=len(futures), desc="Descargando"):
        pass
