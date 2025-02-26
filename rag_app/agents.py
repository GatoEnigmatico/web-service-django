from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool
from .rag import search_rag
import os

# 🔹 Definir una herramienta para buscar en RAG (FAISS)
def document_lookup(query):
    results = search_rag(query)
    return "\n".join([doc["content"] for doc in results]) if results else "No se encontraron documentos relevantes."

# 🔹 Crear herramientas para el agente
tools = [
    Tool(
        name="DocumentSearcher",
        func=document_lookup,
        description="Busca información en los documentos almacenados usando RAG.",
    )
]

# 🔥 Crear el Agente con OpenAI y LangChain
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)