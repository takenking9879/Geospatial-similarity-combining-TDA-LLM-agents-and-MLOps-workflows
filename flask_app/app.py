# app.py
#Librerias pagina inicial
import json
from flask import Flask, render_template, jsonify, request
import pandas as pd
#Librerias agentes
from dotenv import load_dotenv
import os
from agents.agents import agent_generation
from agents.agents import middleware_generation
from agents.query_support import clean_text
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
import pandas as pd
from langchain.tools import tool
from typing import List, Dict, Tuple
from agents.analysis_agent.helpers import recomendar_municipios_por_cultivo_1
from agents.analysis_agent.helpers import recomendar_cultivos_por_municipio_1
from agents.analysis_agent.helpers import cultivos_comunes_dict_1
from agents.analysis_agent.helpers import top_municipios_por_cultivo
from agents.analysis_agent.helpers import top_cultivos_por_municipio
from agents.analysis_agent.prompts import rewriter_prompt_generator
from agents.analysis_agent.prompts import analyst_prompt
from langchain.messages import AIMessage
from agents.overview_multi_agent.helpers import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.tools import DuckDuckGoSearchRun
from agents.overview_multi_agent.prompts import context_prompt
from agents.overview_multi_agent.prompts import web_prompt
from agents.overview_multi_agent.prompts import supervisor_prompt
import asyncio
import time

app = Flask(__name__)

# ----------------------------
# Config / paths
# ----------------------------
SIM_PATH = "results/similarity_matrix.parquet"
CONF_PATH = "results/confianza_matrix.parquet"
GEOJSON_PATH = "results/geojson_light.json"
STATE_PATH = "results/state_to_muns.json"

print("🚀 Iniciando app...")

# --- Cargar matrices ---
df_similarity = pd.read_parquet(SIM_PATH)
df_confidence = pd.read_parquet(CONF_PATH)
df_similarity.index = df_similarity.index.astype(str)
df_similarity.columns = df_similarity.columns.astype(str)
df_confidence.index = df_confidence.index.astype(str)
df_confidence.columns = df_confidence.columns.astype(str)

# --- Cargar objetos precalculados ---
with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    geojson_obj = json.load(f)
with open(STATE_PATH, "r", encoding="utf-8") as f:
    state_to_muns = json.load(f)

# --- Extraer metadatos básicos ---
states = sorted(state_to_muns.keys())
cvegeo_order = [feat["properties"]["CVEGEO"] for feat in geojson_obj["features"]]

print("✅ Datos cargados para página inicial. Estados:", len(states))
print("Cargando archivos de Agentes...")

# Cargar variables del archivo .env
load_dotenv()

# === Claves principales ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# === LangSmith config ===
LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2", "true")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "RAG-Municipios")

os.environ["LANGSMITH_TRACING_V2"] = LANGSMITH_TRACING_V2
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
# --------------------------------------------------
# Agente Analista
# --------------------------------------------------
with open("results/lista_cultivos.json", "r", encoding="utf-8") as f:
    lista_cultivos = json.load(f)

#df_similarity = pd.read_parquet("results/similarity_matrix.parquet")
#df_confidence = pd.read_parquet("results/confianza_matrix.parquet")
suelo = pd.read_parquet("results/soil.parquet")
with open("results/municipios.json", "r", encoding="utf-8") as f:
    municipios_data = json.load(f)
with open("results/cultivos.json", "r", encoding="utf-8") as f:
    cultivo_data = json.load(f) 
with open("results/dict_cvegeo.json", "r", encoding="utf-8") as f:
    dict_cvegeo = json.load(f)


@tool 
def recomendar_municipios_por_cultivo(cultivo: str, min_value: float = 4.0, top_k=3, min_score=0.70) -> dict:
    """
    Given a crop, identifies municipalities with similar agro-environmental conditions 
    suitable for its cultivation.  
    Focuses on finding **new potential areas (similares)** while also listing 
    **current producing municipalities (originales)** and how well they perform.  
    Uses a score = similarity*confidence and includes detailed soil feature vectors for each.

    Parameters:
      • cultivo — Crop name to analyze and find suitable municipalities for.  
      • min_value — Minimum cultivation value to consider a municipality as an “original” producer (default: 4.0).  
      • top_k — Number of top similar municipalities to include in the recommendation (default: 3).  
                 Helps prevent cases where no similar municipality meets the conditions.  
      • min_score — Minimum similarity-confidence score to keep a municipality as valid (default: 0.70).  
    """
    return recomendar_municipios_por_cultivo_1(cultivo_data, df_similarity, df_confidence, cultivo, suelo, min_value, top_k, min_score)

@tool
def recomendar_cultivos_por_municipio(cvegeo: str, top_n: int = 1, min_value: float = 4.0) -> dict:
    """
    Given a municipality (CVEGEO), identifies the most similar municipalities based on 
    similarity*confidence scores.  
    For each similar municipality, it provides:
      • **New crops** that are promising but not currently cultivated.  
      • **Shared crops** with their respective values in both municipalities and the difference.  
    Also includes detailed soil feature vectors for both the original and similar municipalities.

    Parameters:
      • cvegeo — Unique municipality identifier (CVEGEO) to use as the reference.  
      • top_n — Number of top similar municipalities to include in the analysis (default: 1).  
      • min_value — Minimum cultivation value required for a crop to be considered significant 
                    in the similarity and recommendation process (default: 4.0).  
    """
    return recomendar_cultivos_por_municipio_1(cvegeo, df_similarity, df_confidence, municipios_data, suelo, top_n, min_value)

@tool
def cultivos_comunes(cvegeo1: str, cvegeo2: str) -> dict:
    """
    Compares two municipalities (CVEGEO1 and CVEGEO2) and returns the crops they share in common,
    including their respective values and the performance difference.
    Numeric suitability values are mapped to qualitative labels (e.g., “Excelente”, “Regular”, “Malo”) for easier interpretation.
    """
    return cultivos_comunes_dict_1(municipios_data, cvegeo1, cvegeo2)

# ===========================
# Tool 1: Top municipios por cultivo
# ===========================
@tool
def top_municipios_cultivo(cultivo_name: str, N: int = 5) -> List[Dict]:
    """
    Devuelve los top N municipios para un cultivo específico según su valor.
    Nota: Solo devuelve datos de cultivo, no incluye interpretación ni inferencia del modelo.

    Ejemplo:
        top_municipios_cultivo("Manzana", 3)
        # Devuelve los 3 municipios con mayor producción o valor de cultivo "Manzana"

    Args:
        cultivo_name (str): Nombre del cultivo (ej. "Manzana")
        N (int): Número de top resultados a devolver (por defecto 5)

    Returns:
        list of dict: Top N municipios con su información y valor
    """
    return top_municipios_por_cultivo(cultivo_data, cultivo_name, N)

# ===========================
# Tool 2: Top cultivos por municipio
# ===========================
@tool
def top_cultivos_municipio(municipio_id: str, N: int = 5) -> List[Tuple[str, float]]:
    """
    Devuelve los top N cultivos para un municipio específico según su valor.
    Nota: Solo devuelve datos de cultivo, no incluye interpretación ni inferencia del modelo.

    Ejemplo:
        top_cultivos_municipio("01002", 3)
        # Devuelve los 3 cultivos más importantes en el municipio con CVEGEO "01002"

    Args:
        municipio_id (str): CVEGEO del municipio (ej. "01002")
        N (int): Número de top cultivos a devolver (por defecto 5)

    Returns:
        list of tuples: [(nombre_cultivo, valor), ...] ordenados de mayor a menor
    """
    return top_cultivos_por_municipio(municipios_data, municipio_id, N)
rewriter_prompt = rewriter_prompt_generator(lista_cultivos, dict_cvegeo)
rewriter_agent = agent_generation(prompt=rewriter_prompt,
                                  model='gpt-5-mini',
                                  reasoning_effort='minimal'
                                  )#gpt-5-nano, low para debuggin
checkpointer_analyst = InMemorySaver()
middleware_analyst = middleware_generation(messages_to_keep=3)
analyst_agent = agent_generation(prompt=analyst_prompt,
                                model='gpt-5-mini',
                                reasoning_effort='low',
                                tools=[recomendar_municipios_por_cultivo,
                                       recomendar_cultivos_por_municipio,
                                       cultivos_comunes, top_municipios_cultivo,
                                       top_cultivos_municipio],
                                checkpointer=checkpointer_analyst,
                                middleware=[middleware_analyst]
                                )#gpt-5-nano, medium para debuggin
def analyst_chat(query):
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    input_data = {"messages": [{"role": "user", "content": clean_text(query)}]}

    response = rewriter_agent.invoke(input_data)
    ai_content = next(
    (msg.content for msg in response['messages'] if isinstance(msg, AIMessage)),
    None)
    response_rewriter = ai_content
    #print(response_rewriter)
    for token, metadata in analyst_agent.stream(
        {"messages": [{"role": "user", "content": response_rewriter}]},
        stream_mode="messages", config=config
    ):
        print(token.content, end="", flush=True)
# --------------------------------------------------
# Agente Supervisor
# --------------------------------------------------
embedding = download_embeddings()
# embbeded each chunk and upsert the embeddings into your pinecone index
index_name = "research-assistant"
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)
print("✅Archivos Cargados de Agentes")

# RAG tool
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information from internal documentation or PDFs.

    Use this tool when the user asks about technical or implementation details
    within the project. It performs a semantic similarity search in the
    document embeddings to extract the most relevant passages.

    Example queries:
    - "Which library was used for VietorisRips?"
    - "Was DVC used for reproducibility?"
    - "Explain how GPU support was integrated with giotto-tda."
    """
    retrieved_docs = vector_store.similarity_search(query, k=2)
    if not retrieved_docs:
        return "No relevant context found in the documentation.", []
    serialized = "\n\n".join(
        f"📘 **Source:** {doc.metadata}\n{doc.page_content}" for doc in retrieved_docs
    )
    return serialized, retrieved_docs
# DuckDuckGo tool
@tool
def duckduckgo_search(query: str) -> str:
    """Search the web using DuckDuckGo.

    Use this tool for general information retrieval from online sources such as news,
    research sites, and Wikipedia. Limit results to the top 3 most relevant hits.

    Example queries:
    - "Recent research on persistent homology"
    - "Farms cultivating lettuce near Aguascalientes"
    """
    results = DuckDuckGoSearchRun().run(query)
    if not results:
        return "No results found."
    top_results = results[:3]
    return "\n".join(f"🔗 {r['title']}: {r['href']}" for r in top_results)

# Crear el client con Taviliy MCP
client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": "https://mcp.tavily.com/mcp",
            "headers": {"Authorization": f"Bearer {TAVILY_API_KEY}"}
        }
    }
)

def load_mcp_tools(client):
    return asyncio.run(client.get_tools())
# Obtener todas las tools
tools_mcp = load_mcp_tools(client=client)

# Context Agent
context_agent = agent_generation(prompt=context_prompt,
                 model="gpt-5-mini",
                 reasoning_effort="minimal",
                 tools=[retrieve_context])  #gpt-5-nano, low para debuggin
# Web Agent
web_agent = agent_generation(prompt=web_prompt,
                             model="gpt-5-nano",
                             reasoning_effort="low",
                             tools=[duckduckgo_search]+tools_mcp)
# Agents as tools
@tool
def context_retriever(request: str) -> str:
    """Retrieve answers from the project documentation."""
    result = context_agent.invoke({"messages": [{"role": "user", "content": request}]})
    response = result["messages"][-1] if isinstance(result, dict) else result
    return response.content if hasattr(response, "content") else str(response)


@tool
def web_retriever(request: str) -> str:
    """Retrieve information from the web using Tavily or DuckDuckGo."""
    result = web_agent.invoke({"messages": [{"role": "user", "content": request}]})
    response = result["messages"][-1] if isinstance(result, dict) else result
    return response.content if hasattr(response, "content") else str(response)

# Creation of Supervisor Agent
checkpointer_supervisor = InMemorySaver()
middleware_supervisor = middleware_generation(messages_to_keep=2)
supervisor_agent = agent_generation(prompt=supervisor_prompt,
                                model='gpt-5-mini',
                                reasoning_effort='low',
                                tools=[context_retriever,
                                       web_retriever
                                       ],
                                checkpointer=checkpointer_supervisor,
                                middleware=[middleware_supervisor]
                                )#gpt-5-nano, medium para debuggin
def supervisor_chat(query):
    config: RunnableConfig = {"configurable": {"thread_id": "2"}}
    query_cleaned = clean_text(query) 
    #print(response_rewriter)
    for token, metadata in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": query_cleaned}]},
        stream_mode="messages", config=config
    ):
        print(token.content, end="", flush=True)

# --- NUEVO: imports necesarios para las nuevas rutas y streaming
from flask import Response, send_from_directory, send_file
import itertools
from typing import Iterable

# --- NUEVO: Helper generator para stream (usa los agentes ya definidos arriba)
def _stream_agent_response(agent, messages, config):
    """
    NUEVO: generator que itera sobre agent.stream(...) y yield-a cada chunk completo.
    """
    try:
        for item in agent.stream({"messages": messages}, config=config):
            val = item.get("model")
            if val is not None:
                # yield cada mensaje completo del agente
                yield val.get("messages")[0].content
    except Exception as e:
        yield f"\n\n[STREAM ERROR] {str(e)}"

print("✅Se ha cargado todo lo necesario para la app")
# ----------------------------
# Endpoints
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html", states=states)

@app.route("/geojson")
def geojson():
    return jsonify(geojson_obj)

@app.route("/states")
def get_states():
    return jsonify(states)

@app.route("/municipalities")
def get_municipalities():
    state = request.args.get("state")
    return jsonify(state_to_muns.get(state, []))

@app.route("/similarity/<cvegeo>")
def get_similarity_column(cvegeo):
    c = str(cvegeo)
    if c not in df_similarity.columns:
        return jsonify({"error": "CVEGEO not found"}), 404
    s = df_similarity[c].reindex(cvegeo_order)
    values = [None if pd.isna(x) else float(x) for x in s.values]
    return jsonify({"cvegeo_order": cvegeo_order, "values": values})

@app.route("/info/<cvegeo>")
def info_municipio(cvegeo):
    c = str(cvegeo)
    if c not in df_similarity.columns:
        return jsonify({})
    return jsonify({
        "CVEGEO": c,
        "similarity_values": {str(idx): (None if pd.isna(v) else float(v)) for idx, v in df_similarity[c].items()},
        "confidence_values": {str(idx): (None if pd.isna(v) else float(v)) for idx, v in df_confidence.get(c, {}).items()}
    })

# --- NUEVO: Endpoint para stream de Analista (devuelve chunked response)
@app.route("/api/stream/analyst", methods=["POST"])
def api_stream_analyst():
    payload = request.get_json(force=True)
    user_message = payload.get("message", "")
    config = {"configurable": {"thread_id": "1"}}
    messages = [{"role": "user", "content": clean_text(user_message)}]

    # Paso del rewriter
    response = rewriter_agent.invoke({"messages": messages})
    ai_content = next(
        (msg.content for msg in response['messages'] if isinstance(msg, AIMessage)),
        None
    )
    response_rewriter = ai_content

    def generate():
        yield ""  # primer yield para iniciar el response en el cliente
        for chunk in _stream_agent_response(analyst_agent, [{"role": "user", "content": response_rewriter}], config):
            # char por char para efecto ChatGPT
            for c in chunk:
                yield c
                time.sleep(0.01)

    return Response(generate(), mimetype="text/plain; charset=utf-8")

# --- NUEVO: Endpoint para stream de Supervisor (devuelve chunked response)
@app.route("/api/stream/supervisor", methods=["POST"])
def api_stream_supervisor():
    payload = request.get_json(force=True)
    user_message = payload.get("message", "")
    config = {"configurable": {"thread_id": "2"}}
    messages = [{"role": "user", "content": clean_text(user_message)}]

    def generate():
        yield ""  # primer yield para iniciar el stream
        for chunk in _stream_agent_response(supervisor_agent, messages, config):
            for c in chunk:
                yield c

    return Response(generate(), mimetype="text/plain; charset=utf-8")

# --- NUEVO: Endpoint para devolver dict_cvegeo organizado por estado (cliente usa para dropdowns)
@app.route("/api/dict_cvegeo_states")
def api_dict_cvegeo_states():
    """
    NUEVO: Devuelve { "states": [...], "mapping": { "Estado": [ { "cvegeo": "01002", "name": "Asientos" }, ... ] } }
    Se construye a partir de dict_cvegeo (formato: {'01002': 'Asientos, Aguascalientes', ...})
    """
    mapping = {}
    for cve, full in dict_cvegeo.items():
        # Se espera formato "Municipio, Estado"
        parts = [p.strip() for p in full.split(",")]
        if len(parts) >= 2:
            municipio = ", ".join(parts[:-1]).strip()
            estado = parts[-1].strip()
        else:
            # fallback: poner todo en municipio y estado 'Desconocido'
            municipio = full
            estado = "Desconocido"
        mapping.setdefault(estado, []).append({"cvegeo": cve, "name": municipio})
    states_list = sorted(mapping.keys())
    return jsonify({"states": states_list, "mapping": mapping})

# --- NUEVO: Endpoint para devolver lista de cultivos
@app.route("/api/lista_cultivos")
def api_lista_cultivos():
    """
    NUEVO: Devuelve lista_cultivos tal cual (para el dropdown de ayuda al usuario).
    """
    return jsonify(lista_cultivos)

# --- NUEVO: Endpoint para servir PDF solicitado por la página de Revisión (embebido en iframe)
@app.route("/pdf/analysis")
def pdf_analysis():
    """
    NUEVO: Sirve el PDF en data_RAG/... para que el iframe lo muestre.
    Ajusta la ruta si tu archivo está en otro sitio.
    """
    pdf_path = "data_RAG/Análisis Topológico con GPU y Visualización Interactiva Basada en GenAI.pdf"
    # Usamos send_file para manejar nombres con espacios/acento; asegurarnos que el archivo exista.
    return send_file(pdf_path, mimetype="application/pdf")

# --- NUEVO: Ruta para la página del Agente Analista (interfaz)
@app.route("/chat/analyst")
def page_analyst_chat():
    """
    NUEVO: Página dedicada al Agente Analista con dropdowns (estado, municipio, cultivos),
    streaming en tiempo real y diseño futurista.
    """
    return render_template("analyst_chat.html")

# --- NUEVO: Ruta para la página del Agente Supervisor / Revisión Documentación
@app.route("/chat/supervisor")
def page_supervisor_chat():
    """
    NUEVO: Página para revisar documentación (PDF embebido) y chatear con el Agente Supervisor.
    """
    return render_template("supervisor_chat.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
