from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)
parser = StrOutputParser()

# 1. Primera cadena: genera una explicación del concepto
prompt_explicacion = PromptTemplate(
    template="Explica en un párrafo, en español de España, qué es {concepto}.",
    input_variables=["concepto"],
)
cadena_explicacion = prompt_explicacion | llm | parser

print('Ejemplo de explicación "larga" de un concepto:')
print(cadena_explicacion.invoke({"concepto": "el aprendizaje por refuerzo"}))

# 2. Segunda cadena: resume el texto recibido en una sola frase sencilla
prompt_resumen = PromptTemplate(
    template="Resume el siguiente texto en una sola frase apta para principiantes:\n\n{texto}",
    input_variables=["texto"],
)
cadena_resumen = prompt_resumen | llm | parser

# 3. Encadenamos ambas cadenas. La salida de la primera (una cadena de texto)
#    se adapta a la entrada de la segunda, que espera la variable {texto}.
pipeline = cadena_explicacion | (lambda texto: {"texto": texto}) | cadena_resumen

# 4. Ejecutamos el pipeline completo
resultado = pipeline.invoke({"concepto": "el aprendizaje por refuerzo"})

print('\n\nEjemplo de explicación "resumida" de un concepto:')
print(resultado)
