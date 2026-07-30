from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Instanciamos el modelo
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)

# 2. Definimos la plantilla con dos placeholders: {tema} y {pregunta}
prompt = PromptTemplate(
    template="""
Eres un asistente especializado en {tema}.
Reglas:
- Responde en español de España.
- Explica los conceptos de forma simple.
- Usa analogías cuando sea posible.
- Las respuestas deben ser breves, de no más de 100 palabras.

Pregunta: {pregunta}
""",
    input_variables=["tema", "pregunta"],
)

# 3. Conectamos la plantilla con el modelo mediante LCEL (prompt -> llm)
chain = prompt | llm

# 4. Ejecutamos el pipeline proporcionando valores para los placeholders
respuesta = chain.invoke({"tema": "Machine Learning", 
                          "pregunta": "¿Qué es el overfitting?"})

print(respuesta.text)