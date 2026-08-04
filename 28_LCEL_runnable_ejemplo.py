from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Definimos un Runnable propio: pasa el texto a minúsculas
#    y sustituye los espacios en blanco por guiones bajos
a_slug = RunnableLambda(lambda texto: texto.lower().replace(" ", "_"))

# 2. Lo probamos de forma aislada para ver qué hace
print(a_slug.invoke("Hola Mundo Feliz"))

# 3. Ahora lo aplicamos al final de un chain que genera texto con el modelo
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
prompt = PromptTemplate(
    template="Responde en una sola frase breve a la siguiente pregunta: {pregunta}",
    input_variables=["pregunta"],
)

# El Runnable a_slug se encadena después del parser: prompt -> llm -> parser -> a_slug
chain = prompt | llm | StrOutputParser() | a_slug

# 4. Ejecutamos el chain completo
print(chain.invoke({"pregunta": "¿Cuándo pisó el hombre la Luna por primera vez?"}))