from pathlib import Path


# 1. Las rutas parten de esta carpeta, independientemente de dónde ejecutemos.
DIRECTORIO = Path(__file__).resolve().parent
CORPUS_PATH = DIRECTORIO / "corpus"
CHROMA_DB_PATH = DIRECTORIO / "chroma_db"
COLLECTION_NAME = "ciberseguridad"

# 2. Utilizamos Gemini, como en los ejemplos 24 y 34.
# Embeddings y chat utilizan GOOGLE_API_KEY, sin guardar claves aquí.
# Si cambiamos el modelo de embeddings, debemos ejecutar de nuevo el indexador.
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-flash-latest"
TEMPERATURE = 0.1

# 3. Configuración de la fragmentación y de la recuperación.
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 1000
SEARCH_K = 2

# 4. True: usa la conversación; False: trata cada pregunta de forma aislada.
HISTORY = True
PREVIEW_CHARS = 250
