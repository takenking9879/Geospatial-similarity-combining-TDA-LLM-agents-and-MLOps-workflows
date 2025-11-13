from langchain_huggingface import HuggingFaceEmbeddings
#Download the Embeddings from HuggingFace 

def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-distilroberta-v1"

    # Intentar importar torch (opcional)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"  # Si torch no está instalado, usa CPU por defecto

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device}
    )
    return embeddings