import os
import threading
import logging
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log current working directory
current_directory = os.getcwd()
logging.info("Current working directory: %s", current_directory)

INDEX_PATH = "faiss_index"
embeddings = OpenAIEmbeddings()
db_lock = threading.Lock()

def initialize_db():
    logging.info("Initializing FAISS database...")
    if os.path.exists(INDEX_PATH):
        try:
            logging.info("Loading existing FAISS index from %s", INDEX_PATH)
            return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            logging.error("Failed to load FAISS index: %s", str(e))
            return None
    else:
        logging.info("FAISS index not found. Creating a new one...")
        try:
            # Adding sample documents
            texts = ["This is a test document."]
            metadatas = [{"title": "Example"}]
            db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            db.save_local(INDEX_PATH)
            logging.info("New FAISS index created and saved successfully.")
            return db
        except Exception as e:
            logging.error("Error while creating FAISS index: %s", str(e))
            return None

db = initialize_db()

def add_document(content, metadata):
    if db is None:
        logging.error("Cannot add document. FAISS database is not initialized.")
        return
    
    with db_lock:
        try:
            logging.info("Adding new document to FAISS index...")
            db.add_texts([content], metadatas=[metadata])
            db.save_local(INDEX_PATH)
            logging.info("Document added successfully.")
        except Exception as e:
            logging.error("Failed to add document: %s", str(e))

def search(query):
    if db is None:
        logging.error("Cannot perform search. FAISS database is not initialized.")
        return []
    
    with db_lock:
        try:
            logging.info("Performing search for query: %s", query)
            results = db.similarity_search(query)
            logging.info("Search completed. Found %d results.", len(results))
            return [{"content": res.page_content, "metadata": res.metadata} for res in results]
        except Exception as e:
            logging.error("Search failed: %s", str(e))
            return []
