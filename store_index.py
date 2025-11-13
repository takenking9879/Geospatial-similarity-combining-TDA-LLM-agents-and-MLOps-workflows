from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split_token
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from pinecone import Pinecone
from src.utils import create_logger

os.makedirs("logs", exist_ok=True)  # exist_ok=True evita error si ya existe
logger = create_logger("vector_store", "logs/vector_store.log")

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_AdPI_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

logger.info("Creando Vector Store en Pinecone...")
extracted_data = load_pdf_files("flask_app/data_RAG")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split_token(minimal_docs)
embedding = download_embeddings()

index_name = "research-assistant"

pc = Pinecone()
# create the index if it doesn't exist
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=768,  # replace with your embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
logger.info(f"Vector Store {index_name} ha sido creada")

# connect to the index
index = pc.Index(index_name)

PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name= index_name
)
logger.info(f"Se ha subido los datos del RAG a {index_name} en Pinecone")