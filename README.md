# MUIA_PLNA — Procesamiento del Lenguaje Natural

Repositorio de materiales didácticos de la asignatura **Procesamiento del Lenguaje Natural** del Máster Universitario en Inteligencia Artificial (MUIA) de U-tad.

---

## Contenido

### Notebooks

| Notebook | Descripción |
|---|---|
| `01_instalacion_y_prueba_spacy.ipynb` | Verificación del entorno y primera prueba de spaCy |
| `02_primer_ejemplo_spacy.ipynb` | Pipeline básico de spaCy sobre una frase de ejemplo |
| `03_normalizacion_texto_caso_practico.ipynb` | Caso práctico de normalización textual paso a paso |
| `04_actividad_practica_normalizacion.ipynb` | Actividad práctica: construcción de un pipeline de normalización |
| `05_BoW_Tfidf.ipynb` | Vectorización de texto: Bag of Words y TF-IDF con scikit-learn y Gensim |
| `06_metricas_clasificacion.ipynb` | Métricas de evaluación para modelos de clasificación: Accuracy, Precision, Recall, F-Score y ROC-AUC |
| `07_clasificacion_textos_ejemplo.ipynb` | Pipeline completo de clasificación de textos: normalización, BoW, entrenamiento y evaluación de cuatro clasificadores |
| `08_inferencia_modelo_clasificacion_textos.ipynb` | Inferencia sobre un texto nuevo con el vectorizador y el modelo exportados en el notebook 07 |
| `09_actividad_practica_clasificacion_textos.ipynb` | Actividad práctica: pipeline de clasificación de textos sobre un dataset de soporte telco |
| `10_LSI_ejemplo_basico.ipynb` | Latent Semantic Index/Analysis (LSI): teoría y ejemplo básico con Gensim |
| `11_LDA_ejemplo_basico.ipynb` | Latent Dirichlet Allocation (LDA): teoría y ejemplo básico con Gensim |
| `12_LDA_Topic_Modeling_ejemplo_noticias.ipynb` | Topic Modeling con LDA sobre un corpus de noticias: selección del número óptimo de temas mediante coherencia |
| `13_Ejemplo_embeddings_preentrenados.ipynb` | Embeddings preentrenados (NNLM) con TensorFlow Hub: vectorización de palabras y similitud semántica |
| `14_busqueda_semantica_embeddings.ipynb` | Búsqueda semántica con embeddings preentrenados: normalización ligera, comparación de modelos de 50 y 128 dimensiones |
| `15_MLP_BoW_clasificacion_textos.ipynb` | Clasificación de sentimiento de críticas de cine con un Perceptrón Multicapa (TensorFlow-Keras) sobre Bag of Words |
| `16_MLP_Embeddings_clasificacion_textos.ipynb`<br>🧪 [Abrir en Colab](https://colab.research.google.com/drive/1g34yqjNK-Pet16dTlVQjf0Zu6IWGwxz7) | Mismo problema que el notebook 15, sustituyendo Bag of Words por un embedding preentrenado (`nnlm-es-dim128`) como primera capa de la red |
| `17_LSTM_Embeddings_clasificacion_textos.ipynb`<br>🧪 [Abrir en Colab](https://colab.research.google.com/drive/1xvEpuIhwwjGBRRG02Ujbr59hnBpfMAOr) | Mismo problema que los notebooks 15 y 16, sustituyendo el MLP por una LSTM bidireccional para aprovechar el orden de las palabras, con una matriz de embeddings por palabra construida a partir de `nnlm-es-dim128` |
| `18_Transformer_Embeddings_clasificacion_textos.ipynb`<br>🧪 [Abrir en Colab](https://colab.research.google.com/drive/18KO-Fn3fJIfwhP62Ef3-J00lrbLsY--C) | Mismo problema que los notebooks 15, 16 y 17, sustituyendo la LSTM por bloques Transformer construidos desde cero (codificación posicional, autoatención multicabeza y red *feed-forward*) sobre la misma matriz de embeddings por palabra de `nnlm-es-dim128` |
| `20_BERT_BETO_clasificacion_textos.ipynb`<br>🧪 [Abrir en Colab](https://colab.research.google.com/drive/1qS9JBbYP6rbdH7KRJTqpG_UxDuGOP-oB) | Mismo problema que los notebooks 15-18, usando el modelo preentrenado BERT en español BETO (`bert-base-spanish-wwm-cased`) como extractor de características congelado y una cabeza clasificadora entrenada sobre el vector de resumen (`[CLS]`) de 768 dimensiones |
| `21_BERT_BETO_NER_tecnicismos_futbol.ipynb`<br>🧪 [Abrir en Colab](https://colab.research.google.com/drive/1zqDCBqrbbA1ayyWf--74P_HLhv1PqhC7) | Reconocimiento de Entidades Nombradas (NER) con BETO: clasificación palabra a palabra (esquema BIO) para detectar tecnicismos futbolísticos, con una prueba final de generalización a tecnicismos no vistos durante el entrenamiento |

### Scripts
| Script | Descripción |
|---|---|
| `23_request_API_LLM.py` | Primera petición a un LLM (Gemini, vía `langchain-google-genai`): instanciación del modelo y llamada simple con `invoke` |
| `24_chatLLM.py` | Chat con historial gestionado a mano: bucle de conversación por consola que mantiene el contexto turno a turno con `SystemMessage`, `HumanMessage` y `AIMessage` |
| `25_chatLLM_Tool.py` | Chat con *tool calling*: el modelo decide cuándo llamar a una herramienta de búsqueda real en internet (Tavily) para responder preguntas sobre información actual, en lugar de inventar la respuesta |
| `26_PromptTemplate_ejemplo.py` | Plantilla de prompt (`PromptTemplate`) con los placeholders `{tema}` y `{pregunta}`, conectada al modelo mediante LCEL (`prompt \| llm`) |
| `27_ChatPromptTemplate_ejemplo.py` | Plantilla de chat (`ChatPromptTemplate`) con mensaje de sistema, ejemplos *few-shot* en español rioplatense y una pregunta final, conectada al modelo mediante LCEL |
| `28_LCEL_runnable_ejemplo.py` | `RunnableLambda` propio (pasa un texto a minúsculas y sustituye los espacios por guiones bajos), probado de forma aislada y encadenado al final de un chain (`prompt \| llm \| StrOutputParser() \| runnable`) |
| `29_LCEL_composicion_chains_ejemplo.py` | Composición de dos chains independientes (una genera una explicación, otra la resume) encadenadas con una función lambda que adapta la salida de texto de la primera a la variable de entrada que espera la segunda |
| `30_Pydantic_ejemplo.py` | Extracción de datos estructurados con `with_structured_output` y modelos Pydantic (`Portatil`, `Catalogo`): a partir del HTML en bruto de una página de catálogo de portátiles, el modelo genera y valida un listado con marca, modelo, procesador, RAM, disco y precio, que se imprime y se guarda en `portátiles.json` |


---

## Instalación del entorno

> **Requisito previo:** todos los comandos deben ejecutarse desde la carpeta raíz del repositorio. Si acabas de clonar el proyecto, navega hasta ella antes de continuar:
> ```bash
> cd MUIA_PNLA
> ```

Los pasos siguientes utilizan [**uv**](https://github.com/astral-sh/uv), una herramienta moderna y rápida para gestionar entornos e instalar dependencias en Python.

### 1. Instalar uv

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Crear el entorno virtual

Desde la raíz del repositorio:

```bash
uv venv .venv --python 3.11
```

### 3. Activar el entorno

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

### 4. Instalar las dependencias

```bash
uv pip install -r requirements.txt
```

### 5. Descargar los modelos de spaCy

```bash
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

### 6. Registrar el entorno como kernel de Jupyter

```bash
python -m ipykernel install --user --name .venv --display-name "Python (PNLA)"
```

### 7. Abrir Jupyter Notebook

```bash
jupyter notebook
```

Si da algún error en Windows, abrir el Notebook con el siguiente comando:

```bash
python -m jupyter notebook
```

Al crear o abrir un notebook, selecciona el kernel **Python (PNLA)** para asegurarte de que el entorno con spaCy está activo.

---

## Requisitos

- Python 3.11
- uv >= 0.4

Las dependencias Python están declaradas en [`requirements.txt`](requirements.txt).
