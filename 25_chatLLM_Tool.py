'''
Antes de ejecutar este script, asegúrate de instalar las dependencias:
uv pip install langchain_tavily

Tienes que obtener una API key de Tavily (https://app.tavily.com/) y exportarla como variable de entorno:
macOS/Linux: export TAVILY_API_KEY="tu_api_key_aqui"
Windows (PowerShell): setx TAVILY_API_KEY "tu_api_key_aqui"
'''

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# 1. Herramienta de búsqueda real en internet
buscador = TavilySearch(max_results=3)

# 2. Instanciamos el modelo y le "damos" la herramienta con bind_tools
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
llm_con_tools = llm.bind_tools([buscador])

# 3. Diccionario para localizar la herramienta por su nombre al ejecutarla
herramientas = {"tavily_search": buscador}

# 4. Historial: un mensaje de sistema le indica cuándo debe buscar
historial = [
    SystemMessage(content=(
        "Eres un asistente que responde en español de España de forma breve y clara. "
        "Si la pregunta requiere información actual o que no conoces con seguridad, "
        "utiliza la herramienta de búsqueda en lugar de inventar la respuesta. "
        "Al formular la búsqueda, escribe la query como una pregunta completa y específica "
        "que incluya el dato exacto (como 'ganador') que necesitas en lugar de usar solo un "
        "par de palabras clave genéricas."))
]

print("Escribe 'salir' para terminar la conversación.\n")

while True:
    pregunta = input("Tú: ").strip()
    if pregunta.lower() == "salir":
        break

    # Añadimos la pregunta del usuario al historial
    historial.append(HumanMessage(content=pregunta))

    # Bucle interno: mientras el modelo pida herramientas, las ejecutamos.
    # Cuando ya no pida ninguna, es que tiene la respuesta final.
    while True:
        respuesta = llm_con_tools.invoke(historial)
        historial.append(respuesta)

        # Si el modelo NO ha pedido herramientas, ya tenemos la respuesta
        if not respuesta.tool_calls:
            print("Asistente:", respuesta.text)
            break

        # Si ha pedido una o varias búsquedas, las ejecutamos y le devolvemos el resultado
        for llamada in respuesta.tool_calls:
            print(f"  [El asistente busca en internet: {llamada['args']}]")
            herramienta = herramientas[llamada["name"]]
            resultado = herramienta.invoke(llamada["args"])
            historial.append(ToolMessage(content=str(resultado), tool_call_id=llamada["id"]))