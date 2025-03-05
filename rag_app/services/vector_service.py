from pathlib import Path
import threading
import logging
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the index path using Pathlib
INDEX_PATH = Path("faiss_index")


class FAISSManager:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings()
        self.db_lock = threading.Lock()
        self.db: Optional[FAISS] = self.initialize_db()

    def initialize_db(self) -> Optional[FAISS]:
        logging.info("Initializing FAISS database...")
        if self.index_path.exists():
            try:
                logging.info("Loading existing FAISS index from %s", self.index_path)
                return FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                logging.exception("Error loading the FAISS index")
                return None
        else:
            logging.info("FAISS index not found. Creating a new one...")
            try:
                texts = ["This is a test document."]
                metadatas = [{"title": "Example"}]
                db = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
                db.save_local(str(self.index_path))
                logging.info("New FAISS index created and saved successfully.")
                return db
            except Exception as e:
                logging.exception("Error creating the FAISS index")
                return None

    def add_document(self, content: str, metadata: Dict[str, Any]) -> None:
        if self.db is None:
            logging.error("Cannot add document. FAISS database is not initialized.")
            return

        with self.db_lock:
            try:
                logging.info("Adding new document to the FAISS index...")
                self.db.add_texts([content], metadatas=[metadata])
                self.db.save_local(str(self.index_path))
                logging.info("Document added successfully.")
            except Exception as e:
                logging.exception("Failed to add document")

    def search(self, query: str) -> List[Dict[str, Any]]:
        if self.db is None:
            logging.error("Cannot perform search. FAISS database is not initialized.")
            return []

        with self.db_lock:
            try:
                logging.info("Performing search for query: %s", query)
                results = self.db.similarity_search(query)
                logging.info("Search completed. Found %d results.", len(results))
                return [
                    {"content": res.page_content, "metadata": res.metadata}
                    for res in results
                ]
            except Exception as e:
                logging.exception("Search failed")
                return []


faiss_manager = FAISSManager(INDEX_PATH)

if __name__ == "__main__":
    # Display the current directory using Pathlib
    current_directory = Path.cwd()
    logging.info("Current directory: %s", current_directory)

    faiss_manager = FAISSManager(INDEX_PATH)

    # Example: add a document
    faiss_manager.add_document(
        "New content for the document.", {"title": "New Example"}
    )

    # Example: search
    results = faiss_manager.search("test")
    logging.info("Search results: %s", results)
