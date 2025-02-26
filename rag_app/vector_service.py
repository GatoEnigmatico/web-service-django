# vector_service.py
import os
import threading
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

INDEX_PATH = "faiss_index"
embeddings = OpenAIEmbeddings()
db_lock = threading.Lock()

def initialize_db():
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        # Agregar documentos de prueba
        texts = ["Este es un documento de prueba."]
        metadatas = [{"title": "Ejemplo"}]
        db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        db.save_local(INDEX_PATH)
        return db

db = initialize_db()

def add_document(content, metadata):
    with db_lock:
        db.add_texts([content], metadatas=[metadata])
        db.save_local(INDEX_PATH)

def search(query):
    with db_lock:
        results = db.similarity_search(query)
    return [{"content": res.page_content, "metadata": res.metadata} for res in results]