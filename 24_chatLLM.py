from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# 1. Instanciamos el modelo
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)

# 2. El historial es una lista de mensajes gestionada por el desarrollador.
#    Empezamos con un mensaje de sistema que fija el comportamiento del asistente.
historial = [
    SystemMessage(content="Eres un asistente que responde en español de España de forma breve y clara.")
]

print("Escribe 'salir' para terminar la conversación.\n")

while True:
    pregunta = input("Tú: ").strip()
    if pregunta.lower() == "salir":
        break

    # 3. Añadimos la pregunta del usuario al historial
    historial.append(HumanMessage(content=pregunta))

    # 4. Enviamos TODO el contexto (historial completo) en cada petición
    respuesta = llm.invoke(historial)
    print("Asistente:", respuesta.text)

    # 5. Guardamos la respuesta del modelo en el historial para el siguiente turno
    historial.append(AIMessage(content=respuesta.text))