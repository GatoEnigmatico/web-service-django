from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

# Configura embeddings
embeddings = OpenAIEmbeddings()

# Carga documentos
docs = [
    {"title": "Ejemplo", "content": "Este es un documento de prueba."}
]

# Guarda embeddings en Chroma
db = Chroma.from_texts([doc["content"] for doc in docs], embeddings)

# Función para buscar en RAG
def search_rag(query):
    results = db.similarity_search(query)

    # Extrae solo la información relevante
    formatted_results = []
    for res in results:
        formatted_results.append({
            "content": res.page_content,
            "metadata": res.metadata
        })

    return formatted_results

