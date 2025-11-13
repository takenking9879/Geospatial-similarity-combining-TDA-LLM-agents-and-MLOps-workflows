
import pandas as pd
import numpy as np

# --------------------------------------------------
# Dado un cultivo, dar sugerencias nuevas de donde plantar
# --------------------------------------------------

def municipios_similares_iniciales(cultivos_dict, similarity, confidence, cultivo, min_valor=4, top_k=5, min_score=0.70):
    """
    Devuelve municipios similares distintos para un cultivo, ordenados por score descendente:
    - Solo originales con valor >= min_valor
    - Calcula score = similarity * confidence
    - Considera hasta top_k similares por cada original
    - Mantiene solo el mejor score por municipio similar
    - Indica a qué municipio original se parece
    - Excluye similares que ya tienen valor registrado (solo N/A)
    - Filtra los similares con score < min_score
    """
    if cultivo not in cultivos_dict:
         return {
        "originales": [],
        "similares_distintos": [],
        "mensaje": f"El cultivo '{cultivo}' no se encontró en los datos disponibles."
    }
    
    valores_cultivo = {m["CVEGEO"]: m["valor"] for m in cultivos_dict[cultivo]}

    # Filtrar originales válidos
    originales = [
        {"CVEGEO": m, "value": v}
        for m, v in valores_cultivo.items()
        if v >= min_valor and m in similarity.columns
    ]
    if not originales:
        return {
            "originales": [],
            "similares_distintos": [],
            "mensaje": (
                f"El cultivo '{cultivo}' existe en los datos, pero ningún municipio "
                f"cumple con el valor mínimo requerido (≥ {min_valor})."
            )
        }

    orig_cvegeos = [o["CVEGEO"] for o in originales]
    similares_map = {}

    for m_orig in orig_cvegeos:
        if m_orig not in similarity.index:
            continue

        # Calcular score = similarity * confidence
        sim_scores = (similarity.loc[m_orig] * confidence.loc[m_orig]).drop(m_orig)
        sim_scores_sorted = sim_scores.sort_values(ascending=False).head(top_k)

        for m_sim, score_val in sim_scores_sorted.items():
            sim_val = similarity.loc[m_orig, m_sim]
            conf_val = confidence.loc[m_orig, m_sim]
            valor_sim = valores_cultivo.get(m_sim, "N/A")

            # Solo mantener similares sin valor registrado (N/A)
            if valor_sim != "N/A":
                continue

            # Guardar solo el mejor score para cada similar
            if m_sim not in similares_map or score_val > similares_map[m_sim]["score"]:
                similares_map[m_sim] = {
                    "CVEGEO": m_sim,
                    "similarity": sim_val,
                    "confidence": conf_val,
                    "score": score_val,
                    "value": valor_sim,
                    "similar_to": m_orig
                }

    # Convertir a lista y filtrar por min_score
    similares_distintos = [
        v for v in similares_map.values() if v["score"] >= min_score *100
    ]

    # Ordenar por score descendente
    similares_distintos = sorted(similares_distintos, key=lambda x: x["score"], reverse=True)

    return {
        "originales": originales,
        "similares_distintos": similares_distintos
    }
def recomendar_municipios_por_cultivo_1(cultivos_dict, similarity, confidence, cultivo,
                                        suelo: pd.DataFrame, min_valor=4, top_k=3, min_score=0.70) -> dict:
    """
    Finds distinct municipalities with agro-environmental conditions suitable for a given crop.
    - Considers only original municipalities with value ≥ min_valor.
    - Computes a composite score = similarity * confidence.
    - Keeps only the best match per similar municipality and filters by min_score.
    - Excludes municipalities that already cultivate the crop (only "N/A" values are considered similar).
    - Adds soil feature vectors for both originals and similar municipalities, excluding the redundant 'CVEGEO' column.
    Focuses primarily on identifying **new potential areas for cultivation** while also listing
    **current producing municipalities** and how well they perform.
    """
    
    resultado_cultivo = municipios_similares_iniciales(cultivos_dict, similarity, confidence, cultivo, min_valor, top_k, min_score)

    # Función interna para obtener features de un solo municipio
    def obtener_features(cvegeo: str) -> dict:
        fila = suelo[suelo['CVEGEO'] == cvegeo]
        if fila.empty:
            return None
        features = fila.iloc[0].to_dict()
        features.pop("CVEGEO", None)  # Eliminar CVEGEO si existe
        return features

    # Agregar features a originales
    for orig in resultado_cultivo.get("originales", []):
        cve = orig["CVEGEO"]
        orig["features_suelo"] = obtener_features(cve)

    # Agregar features a similares
    for sim in resultado_cultivo.get("similares_distintos", []):
        cve = sim["CVEGEO"]
        sim["features_suelo"] = obtener_features(cve)

    return resultado_cultivo


# --------------------------------------------------
# Dado un municipio, da sugerencias de donde hay otro similar y que plantar en ese
# --------------------------------------------------

def top_cultivos_recomendados_iniciales(cvegeo, similarity, confidence, municipios_data, top_n=3, min_valor=4.0):
    """
    Devuelve:
    - original: CVEGEO dado
    - similares: lista de los top_n municipios más parecidos con similarity, confidence y score
    - cultivos_recomendados: lista de cultivos con valor en el municipio original >= min_valor
    """
    if cvegeo not in similarity.columns:
        return {"error": f"{cvegeo} no está en similarity columns"}
    
    # Calcular score
    sim_scores = similarity.loc[cvegeo] * confidence.loc[cvegeo]
    sim_scores[cvegeo] = -1  # no contarse a sí mismo
    
    # Obtener los top_n más similares
    top_similares_cvegeos = sim_scores.nlargest(top_n).index.tolist()
    
    # Construir lista de similares con similarity, confidence y score
    similares_list = []
    for sim_cvegeo in top_similares_cvegeos:
        sim_val = similarity.loc[cvegeo, sim_cvegeo]
        conf_val = confidence.loc[cvegeo, sim_cvegeo]
        score_val = sim_val * conf_val
        similares_list.append({
            "CVEGEO": sim_cvegeo,
            "similarity": sim_val,
            "confidence": conf_val,
            "score": score_val
        })
    
    # Obtener cultivos recomendados según el municipio original
    orig_data = next((m for m in municipios_data if m["CVEGEO"] == cvegeo), None)
    cultivos_recomendados = []
    if orig_data:
        for cultivo, valor in orig_data["cultivos"].items():
            if valor >= min_valor:
                cultivos_recomendados.append({"cultivo": cultivo, "value_original": valor})
    
    return {
        "original": cvegeo,
        "similares": similares_list,
        "cultivos_recomendados_inicialmente": cultivos_recomendados
    }

def recomendar_cultivos_por_municipio_1(cvegeo, similarity, confidence, municipios_data, suelo: pd.DataFrame, top_n=3, min_valor=4.0) -> dict:
    """
    Devuelve:
    - original: CVEGEO dado
    - similares: lista de los top_n municipios más parecidos con similarity, confidence y score
    - Para cada similar:
        * cultivos_nuevos: cultivos recomendados que no cultiva actualmente
        * cultivos_compartidos: cultivos comunes con valor_original, valor_similar y diferencia
    Añade los features de suelo al original y a los similares de un resultado de cultivo,
    eliminando la columna 'CVEGEO' de los features para evitar redundancia.
    """
    resultado_cultivo = top_cultivos_recomendados_iniciales(
        cvegeo, similarity, confidence, municipios_data, top_n, min_valor
    )

    # Función interna para obtener features de un solo municipio
    def obtener_features(cvegeo: str) -> dict:
        fila = suelo[suelo['CVEGEO'] == cvegeo]
        if fila.empty:
            return None
        features = fila.iloc[0].to_dict()
        features.pop("CVEGEO", None)  # Eliminar CVEGEO si existe
        return features

    # Agregar features al original
    if "original" in resultado_cultivo:
        cve = resultado_cultivo["original"]
        resultado_cultivo["features_suelo_original"] = obtener_features(cve)

    # Obtener cultivos del original
    orig_data = next((m for m in municipios_data if m["CVEGEO"] == cvegeo), None)
    cultivos_original = orig_data["cultivos"] if orig_data else {}

    # Agregar features y cultivos nuevos/compartidos a los similares
    for sim in resultado_cultivo.get("similares", []):
        cve = sim["CVEGEO"]
        sim["features_suelo"] = obtener_features(cve)

        # Buscar cultivos del similar
        sim_data = next((m for m in municipios_data if m["CVEGEO"] == cve), None)
        if sim_data:
            cultivos_similar = sim_data["cultivos"]
            cultivos_existentes = set(cultivos_similar.keys())

            cultivos_nuevos = []
            cultivos_compartidos = []

            for cultivo, valor_orig in cultivos_original.items():
                if valor_orig >= min_valor:
                    if cultivo in cultivos_existentes:
                        valor_sim = cultivos_similar[cultivo]
                        diferencia = float(valor_sim) - float(valor_orig)
                        cultivos_compartidos.append({
                            "cultivo": cultivo,
                            "value_original": valor_orig,
                            "value_similar": valor_sim,
                            "difference": abs(diferencia)
                        })
                    else:
                        cultivos_nuevos.append({
                            "cultivo": cultivo,
                            "value_original": valor_orig
                        })

            sim["cultivos_nuevos"] = cultivos_nuevos
            sim["cultivos_compartidos"] = cultivos_compartidos

    # Quitar lista intermedia del resultado
    resultado_cultivo.pop("cultivos_recomendados_inicialmente", None)

    return resultado_cultivo


# --------------------------------------------------
# Dado dos municipios, te da los cultivos de ambos con su categoria y compara que tan diferentes son
# --------------------------------------------------

def cultivos_comunes_mapeados(municipios_data, cvegeo1: str, cvegeo2: str) -> dict:
    """
    Returns the common crops between two municipalities and their values,
    using a custom mapping of values to labels.
    
    Args:
        cvegeo1: CVEGEO del primer municipio.
        cvegeo2: CVEGEO del segundo municipio.
    
    Returns:
        dict con cultivos comunes y valores mapeados.
    """
    mapeo = {
    1: 'Muy Malo',
    2: 'Malo',
    3: 'Regular',
    4: 'Bueno',
    5: 'Excelente'
    }
    # Buscar municipios
    m1 = next((m for m in municipios_data if m["CVEGEO"] == cvegeo1), None)
    m2 = next((m for m in municipios_data if m["CVEGEO"] == cvegeo2), None)
    
    if not m1 or not m2:
        return {"error": "Uno o ambos municipios no fueron encontrados."}
    
    cultivos1 = m1.get("cultivos", {})
    cultivos2 = m2.get("cultivos", {})
    
    comunes = set(cultivos1.keys()) & set(cultivos2.keys())
    
    if not comunes:
        return {"mensaje": f"No hay cultivos compartidos entre {cvegeo1} y {cvegeo2}."}
    
    resultado = {
        cultivo: {
            f"{cvegeo1}": f"{cultivos1[cultivo]} ({mapeo.get(cultivos1[cultivo], 'desconocido')})",
            f"{cvegeo2}": f"{cultivos2[cultivo]} ({mapeo.get(cultivos2[cultivo], 'desconocido')})",
            "diferencia": f"{abs(cultivos1[cultivo]-cultivos2[cultivo])}"
        }
        for cultivo in comunes
    }
    
    return resultado

def cultivos_comunes_dict_1(municipios_data, cvegeo1: str, cvegeo2: str) -> dict:
    """
    Devuelve un diccionario con los cultivos comunes entre dos municipios (cvegeo1 y cvegeo2),
    sus valores y la diferencia.

    Args:
        municipios_data: lista de diccionarios con información de los municipios
        cvegeo1: CVEGEO del primer municipio
        cvegeo2: CVEGEO del segundo municipio

    Returns:
        dict con estructura:
        {
            "cultivo1": {cvegeo1: valor1, cvegeo2: valor2, "diferencia": diff},
            "cultivo2": {...},
            ...
        }
    """
    # Usamos tu función existente para obtener los cultivos comunes
    resultado = cultivos_comunes_mapeados(municipios_data, cvegeo1, cvegeo2)
    
    if "error" in resultado or "mensaje" in resultado:
        return {}  # Devuelve dict vacío si no hay resultados

    data = {}
    for cultivo, valores in resultado.items():
        data[cultivo] = {
            cvegeo1: valores.get(cvegeo1, None),
            cvegeo2: valores.get(cvegeo2, None),
            "diferencia": np.float32(valores.get("diferencia", 0.0))
        }
    
    return data

# --------------------------------------------------
# Dado un cultivo, te da municipios donde ya se cultiva y que tan bien les va
# --------------------------------------------------

def top_municipios_por_cultivo(cultivo_data, cultivo_name, N=5):
    """
    Devuelve los top N municipios para un cultivo específico según su valor.
    
    Args:
        cultivo_data (dict): Datos tipo cultivo -> lista de dicts de municipios
        cultivo_name (str): Nombre del cultivo
        N (int): Número de top resultados a devolver
    
    Returns:
        list of dict: Top N municipios con su información y valor
    """
    if cultivo_name not in cultivo_data:
        return []
    
    # Ordenamos por 'valor' descendente
    sorted_list = sorted(cultivo_data[cultivo_name], key=lambda x: x['valor'], reverse=True)
    return sorted_list[:N]

# --------------------------------------------------
# Dado un municipio, te da los mejores cultivos de este
# --------------------------------------------------

def top_cultivos_por_municipio(municipios_data, municipio_id, N=5):
    """
    Devuelve los top N cultivos para un municipio específico según su valor.
    
    Args:
        municipios_data (list of dict): Lista de municipios con sus cultivos
        municipio_id (str): CVEGEO del municipio
        N (int): Número de top cultivos a devolver
    
    Returns:
        list of tuples: [(nombre_cultivo, valor), ...] ordenados de mayor a menor
    """
    # Buscamos el municipio
    municipio = next((m for m in municipios_data if m['CVEGEO'] == municipio_id), None)
    if municipio is None:
        return []
    
    # Obtenemos y ordenamos los cultivos por valor
    sorted_cultivos = sorted(municipio['cultivos'].items(), key=lambda x: x[1], reverse=True)
    return sorted_cultivos[:N]
