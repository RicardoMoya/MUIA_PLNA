from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# HTML descargado previamente de https://www.pccomponentes.com/categorias/portatiles
# y guardado en disco: el sitio bloquea peticiones automatizadas (403), así que
# partimos de un snapshot local en lugar de descargarlo en cada ejecución.
RUTA_HTML = "corpus/portatiles_pccomponentes.html"


# 1. Definimos con Pydantic la estructura que queremos obtener de cada portátil
class Portatil(BaseModel):
    marca: str = Field(description="Marca del portátil, por ejemplo HP o Lenovo")
    modelo: str = Field(description="Modelo del portátil")
    procesador: str = Field(description="Procesador, por ejemplo Intel Core i5-13420H")
    ram: str = Field(description="Memoria RAM, por ejemplo 16 GB")
    disco: str = Field(description="Capacidad y tipo de disco, por ejemplo 512GB SSD")
    precio: str = Field(description="Precio actual, por ejemplo 469,99€")


class Catalogo(BaseModel):
    portatiles: list[Portatil]


# 2. Leemos el HTML de la página, tal cual, sin extraer ni limpiar nada
# El html que se esta en un fichero local se ha obtenido de https://www.pccomponentes.com/categorias/portatiles
with open(RUTA_HTML, encoding="utf-8") as f:
    html = f.read()

# 3. Le pasamos el HTML completo al modelo y que sea él quien encuentre los
#    datos de cada portátil entre todo el ruido de etiquetas, scripts, menús, etc.
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
llm_estructurado = llm.with_structured_output(Catalogo)

prompt = (
    "El siguiente es el HTML completo de una página con un catálogo de portátiles. "
    "Localiza todos los portátiles que aparecen y genera un listado con la marca, "
    "el modelo, el procesador, la RAM, el disco y el precio de cada uno:\n\n" + html
)

catalogo: Catalogo = llm_estructurado.invoke(prompt)

# 4. Mostramos el resultado por pantalla y lo guardamos en un fichero JSON
json_resultado = catalogo.model_dump_json(indent=2)
print(json_resultado)

with open("portátiles.json", "w", encoding="utf-8") as f:
    f.write(json_resultado)
