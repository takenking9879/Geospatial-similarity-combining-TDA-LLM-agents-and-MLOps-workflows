import re

#Limpiador de queries inicial
def clean_text(text: str) -> str:
    """
    Normaliza un string:
    - Convierte a minúsculas
    - Quita símbolos innecesarios (excepto ? y !)
    - Elimina espacios múltiples y caracteres innecesarios
    """
    # Convertir a minúsculas
        
    # Quitar símbolos que no sean letras, números, espacios, ?, !
    text = re.sub(r"[^a-z0-9áéíóúüñ\s?!.,;:%+*'\"()\[\]-]", "", text, flags=re.IGNORECASE)
    
    # Reemplazar múltiples espacios por uno solo
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text