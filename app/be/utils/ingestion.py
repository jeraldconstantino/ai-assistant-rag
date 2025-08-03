from pathlib import Path
import time

# Third-party libraries
from langchain_chroma import Chroma
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from loguru import logger

# Custom libraries
from app.be.core.config import settings

class FileIngestor:

    def __init__(self):
        """Initialize the FileIngestor"""
        self.DATA_PATH = settings.src_data_path
        self.VECTOR_STORE_PATH = settings.vector_store_path
        self.EMBEDDINGS_MODEL = settings.embeddings_model
        self.API_KEY = settings.openai_api_key

    def load_documents(self):
        """Load PDF, DOCX, and TXT documents from the specified directory."""
        logger.info(f"Loading documents from {self.DATA_PATH} path.")
        documents = []
        file_patterns = ["*.pdf", "*.docx", "*.txt"]
        loader_mapping = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader
        }

        for pattern in file_patterns:
            for file_path in Path(self.DATA_PATH).glob(pattern):
                
                # Skip files already marked as ingested
                if file_path.name.endswith(".ingested"):
                    continue

                file_ext = file_path.suffix.lower()
                loader_cls = loader_mapping.get(file_ext)

                if not loader_cls:
                    logger.warning(f"Unknown file type: {file_path.name}. Skipping.")
                    continue 

                try:
                    loader = loader_cls(str(file_path))
                    doc = loader.load()
                    documents.extend(doc)

                    # Rename file to prevent re-ingestion
                    ingested_path = file_path.with_name(file_path.name + ".ingested")
                    file_path.rename(ingested_path)

                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

        return documents
    
    def transform_docs_to_chunks(self, 
                                 documents, 
                                 CHUNK_SIZE=500, 
                                 CHUNK_OVERLAP=200):
        """
        Transform loaded documents into chunks for vector storage.
        
        Args:
            documents (list): List of documents to be chunked.
            CHUNK_SIZE (int): Size of each chunk.
            CHUNK_OVERLAP (int): Overlap size between chunks.
        
        Returns:
            list: List of text chunks."""
        
        logger.info(f"Transforming {len(documents)} documents into chunks with size {CHUNK_SIZE} and overlap {CHUNK_OVERLAP}.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True
        )

        chunks = text_splitter.split_documents(documents)
        logger.info(f"Number of chunks created: {len(chunks)}")
        return chunks

    def save_vector_store(self, chunks):
        """
        Save the vector store to the specified directory.
        
        Args:
            chunks (list): List of text chunks to be saved in the vector store.
        """
        
        logger.info("Converting chunks to embeddings and saving to vector store.")
        start_time = time.time()

        # Ensure no existing Vector Store instance is holding the folder
        if hasattr(self, "VECTOR_STORE") and self.VECTOR_STORE:
            self.VECTOR_STORE._collection = None
            self.VECTOR_STORE = None
        
        embeddings = OpenAIEmbeddings(model=self.EMBEDDINGS_MODEL,
                                      openai_api_key=self.API_KEY)
        Chroma.from_documents(
            chunks, embeddings, persist_directory=self.VECTOR_STORE_PATH
        )
        end_time = time.time()
        run_time = end_time - start_time
        logger.info(f"Vector store created in {run_time:.2f} seconds.")
        logger.info(f"Vector store saved to {self.VECTOR_STORE_PATH}")

    def start_ingestion_session(self):
        """Main method to run the file ingestion process."""
        logger.info("Starting document ingestion...")
        documents = self.load_documents()
        logger.info(f"Loaded {len(documents)} documents from {self.DATA_PATH} path.")
        if not documents:
            logger.warning("No documents found to ingest.")
            return
        
        chunks = self.transform_docs_to_chunks(documents)
        if not chunks:
            logger.warning("No chunks created from the documents.")
            return
        
        self.save_vector_store(chunks)
        logger.info("Document ingestion completed.")