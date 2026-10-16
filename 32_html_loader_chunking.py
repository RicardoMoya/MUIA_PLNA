from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_text_splitters import HTMLHeaderTextSplitter

# 1. Elegimos la fuente del HTML.
# Para cargar la página por HTTP, comentar la línea del fichero local y descomentar esta línea:
html = requests.get("https://www.sansilvestrevallecana.com/reglamento_popular.php", timeout=30).text

# Cargamos el HTML guardado localmente en la carpeta corpus.
# html = Path("corpus/SAN_SILVESTRE_VALLECANA_2026.html").read_text(encoding="utf-8")

# 2. HTMLHeaderTextSplitter detecta los encabezados h2 «Artículo N».
#    Puede devolver el encabezado y el contenido asociado en Documents distintos.
text_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[("h2", "Artículo")],
)
documentos_html = text_splitter.split_text(html)

# 3. Agrupamos los elementos que comparten el mismo encabezado h2.
#    Así el número, el título h3 y el texto del artículo forman un solo Document.
partes_por_articulo = {}
metadatos_por_articulo = {}

for documento in documentos_html:
    articulo = documento.metadata.get("Artículo")
    if articulo is None:
        continue

    texto = documento.page_content.strip()
    if texto:
        partes_por_articulo.setdefault(articulo, []).append(texto)
        metadatos_por_articulo.setdefault(articulo, documento.metadata.copy())

chunks = [
    Document(page_content="\n".join(partes),
             metadata=metadatos_por_articulo[articulo])
    for articulo, partes in partes_por_articulo.items()
]

print(f"Artículos convertidos en chunks: {len(chunks)}")

# 4. Mostramos un único fragmento por artículo.
for numero, chunk in enumerate(chunks, start=1):
    articulo = chunk.metadata.get("Artículo", "Artículo sin título")
    print(f"\n--- Chunk {numero} | {articulo} ---")
    print(chunk.page_content)
