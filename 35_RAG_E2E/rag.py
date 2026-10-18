from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

from config import CHAT_MODEL, CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, \
    TEMPERATURE, SEARCH_K
from prompts import RAG_PROMPT, CONSULTA_PROMPT


def inicializar_rag():
    # 1. Comprobamos que el indexador ha creado la base de datos local.
    if not (CHROMA_DB_PATH / "chroma.sqlite3").is_file():
        raise FileNotFoundError("Primero ejecuta indexador.py para crear la base de datos.")

    # 2. Abrimos la colección sin crear otra si el nombre no coincide.
    # Mismo modelo que en la indexación, ahora para codificar preguntas.
    modelo_embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type="RETRIEVAL_QUERY")
    base_datos = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=modelo_embeddings,
        persist_directory=str(CHROMA_DB_PATH),
        create_collection_if_not_exists=False)
    retriever = base_datos.as_retriever(search_kwargs={"k": SEARCH_K})

    # 3. Conectamos los prompts con Gemini mediante LCEL.
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=TEMPERATURE)
    cadena_consulta = CONSULTA_PROMPT | llm | StrOutputParser()
    cadena_respuesta = RAG_PROMPT | llm | StrOutputParser()
    return retriever, cadena_consulta, cadena_respuesta


def responder(pregunta, historial, retriever, cadena_consulta, cadena_respuesta):
    # 1. Reformulamos solo cuando recibimos un historial no vacío.
    consulta = pregunta
    if historial:
        consulta = cadena_consulta.invoke({
            "pregunta": pregunta, "historial": historial}).strip()
        if not consulta:
            consulta = pregunta

    # 2. Recuperamos UNA sola vez los fragmentos que enviaremos al modelo.
    documentos = retriever.invoke(consulta)
    if not documentos:
        return "No se han recuperado fragmentos para responder a la pregunta.", []

    # 3. Numeramos los fragmentos y añadimos su procedencia al contexto.
    fragmentos = []
    for numero, documento in enumerate(documentos, start=1):
        fuente = documento.metadata.get("source", "No especificada")
        pagina = documento.metadata.get("page", "No especificada")
        if isinstance(pagina, int):
            pagina += 1  # PyPDFLoader numera las páginas desde cero.
        fragmentos.append(
            f"[Fragmento {numero}] - Fuente: {fuente} - Página: {pagina}\n"
            f"{documento.page_content}")
    contexto = "\n\n".join(fragmentos)

    # 4. Devolvemos la respuesta y los mismos documentos enviados al modelo.
    respuesta = cadena_respuesta.invoke({
        "pregunta": pregunta,
        "historial": historial,
        "contexto": contexto})
    return respuesta, documentos
