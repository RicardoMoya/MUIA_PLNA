from langchain_core.messages import HumanMessage, AIMessage

from config import PREVIEW_CHARS
from rag import inicializar_rag, responder


def main():
    # 1. Inicializamos el sistema una sola vez para toda la sesión.
    retriever, cadena_consulta, cadena_respuesta = inicializar_rag()
    historial = []
    print("Asistente docente de ciberseguridad")
    print("Escribe 'salir' para terminar.\n")

    while True:
        pregunta = input("Tú: ").strip()
        if pregunta.lower() == "salir":
            break
        if not pregunta:
            continue

        # 2. Utilizamos siempre el historial para la reformulación y la generación.
        try:
            respuesta, documentos = responder(
                pregunta, historial,
                retriever, cadena_consulta, cadena_respuesta)
        except Exception as error:
            print(f"No se ha podido responder: {error}\n")
            continue

        print(f"\nAsistente: {respuesta}")

        # 3. Mostramos exactamente los fragmentos proporcionados al LLM.
        print("\n" + "-" * 70)
        print("DOCUMENTOS TENIDOS EN CUENTA")
        if not documentos:
            print("No se han enviado fragmentos al LLM.")
        for numero, documento in enumerate(documentos, start=1):
            print("-" * 70)
            print(f"Documento {numero} [Fragmento {numero}]")
            for campo, valor in documento.metadata.items():
                if campo == "page" and isinstance(valor, int):
                    valor += 1
                    campo = "page (numeración desde 1)"
                print(f"{campo}: {valor}")
            print(f"\nPrimeros {PREVIEW_CHARS} caracteres:")
            print(documento.page_content[:PREVIEW_CHARS])
        print("-" * 70 + "\n")

        # 4. Conservamos los mensajes de la sesión, como en el script 24.
        historial.append(HumanMessage(content=pregunta))
        historial.append(AIMessage(content=respuesta))


if __name__ == "__main__":
    main()
