'''
Antes de ejecutar este script, asegúrate de instalar las dependencias:
uv pip install langchain==1.2.6 langchain-google-genai
'''

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)
                             
respuesta = llm.invoke("¿Quién fue la primera persona en recibir más de un premio Nobel?")
print(respuesta.text)
