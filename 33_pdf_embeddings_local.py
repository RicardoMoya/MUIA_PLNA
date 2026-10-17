from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import sys
import io

# Configurar la salida para UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# 1. Cargamos el mismo PDF utilizado en el script 31.
loader = PyPDFLoader("corpus/reglamento-globos-final.pdf")
paginas = loader.load()
print(f"Páginas cargadas: {len(paginas)}")

# 2. Dividimos el texto priorizando los encabezados con letra y punto.
#    Si un bloque supera los 2000 caracteres, se aplican separadores menores.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=0,
    separators=[r"\b[a-zA-Z]*\.\s*.+", "\n\n", "\n", " ", ""],
    is_separator_regex=True,
)
chunks = text_splitter.split_documents(paginas)
print(f"Chunks generados: {len(chunks)}")

# 3. Cargamos BGE-M3 desde Hugging Face y lo ejecutamos en el equipo local.
#    La primera ejecución descarga el modelo y puede tardar varios minutos.
modelo_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 4},
)

# 4. Calculamos un vector para cada chunk.
textos = [chunk.page_content for chunk in chunks]
vectores = modelo_embeddings.embed_documents(textos)

# 5. Mostramos el tamaño del vector y unos pocos valores como ejemplo.
for numero, (chunk, vector) in enumerate(zip(chunks, vectores), start=1):
    pagina = chunk.metadata.get("page", "desconocida")
    pagina = pagina + 1 if isinstance(pagina, int) else pagina

    print(f"\n--- Chunk {numero}")
    print(f"Página: {pagina}")
    contenido = (chunk.page_content[:100].replace('\n', ' '))
    print(f"Contenido del chunk: {contenido}...")
    print(f"Caracteres del chunk: {len(chunk.page_content)}")
    print(f"Dimensiones del embedding: {len(vector)}")
    print(f"Primeros 10 valores: {[round(valor, 4) for valor in vector[:10]]}")