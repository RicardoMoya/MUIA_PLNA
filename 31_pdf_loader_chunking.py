from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Cargamos el PDF. PyPDFLoader devuelve un Document por cada página.
loader = PyPDFLoader("corpus/reglamento-globos-final.pdf")
paginas = loader.load()

print(f"Páginas cargadas: {len(paginas)}")

# 2. Configuramos la fragmentación recursiva.
# [Opción 1: usa los separadores por defecto (párrafos, líneas, espacios y caracteres).
#  Limita los fragmentos a unos 1000 caracteres y repite hasta 200 entre fragmentos.]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)

# [Opción 2: prioriza separar antes de encabezados con una letra y un punto (a., B.).
#  Si hace falta, recurre a párrafos, líneas, espacios y caracteres. Usa fragmentos
#  de hasta unos 2000 caracteres y no repite texto entre fragmentos.]
'''text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=0,
    separators=[r"\b[a-zA-Z]\.\s.+", "\n\n", "\n", " ", ""],
    is_separator_regex=True)
'''
# split_documents conserva metadatos de origen, como la página del PDF.
chunks = text_splitter.split_documents(paginas)

print(f"Chunks generados: {len(chunks)}")

# 3. Mostramos el origen y el contenido de cada fragmento para comprobar
#    si las normas numeradas quedan agrupadas de forma comprensible.
for numero, chunk in enumerate(chunks, start=1):
    pagina = chunk.metadata.get("page", "desconocida")
    inicio = chunk.metadata.get("start_index", "desconocido")

    print(f"\n--- Chunk {numero} | página {pagina + 1 if isinstance(pagina, int) else pagina} | posición {inicio} ---")
    print(chunk.page_content)
