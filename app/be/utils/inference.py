# Third-party libraries
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from loguru import logger
from typing import Optional

# Custom libraries
from app.be.core.config import settings
from app.be.utils.model import invoke_model
from app.be.schemas.inference_models import AIModelParameters

class ModelInference:
    def __init__(self):
        """Initialize the ModelInference class."""
        self.EMBEDDINGS_MODEL = settings.embeddings_model
        self.VECTOR_STORE_PATH = settings.vector_store_path
        self.API_KEY = settings.openai_api_key
        self.PROMPT_TEMPLATE = settings.prompt_template
        self.VECTOR_STORE = self.initiate_vector_store()        
        self.RELEVANCE_THRESHOLD = settings.relevance_threshold

    def initiate_vector_store(self):
        """Initialize the vector store with OpenAI embeddings."""
        logger.info("Initializing vector store...")
        embeddings = OpenAIEmbeddings(model=self.EMBEDDINGS_MODEL,
                                      openai_api_key=self.API_KEY)
        vector_store = Chroma(
            persist_directory=self.VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
        return vector_store
    
    def perform_similarity_search(self, query: str, k: int = 5) -> list:
        """
        Perform similarity search on the vector store.
        
        Args:
            query (str): The query string to search for.
            k (int): The number of results to return (default is 5).
        
        Returns:
            list: List of documents similar to the query.
        """
        logger.info("Performing similarity search...")
        results_with_scores = self.VECTOR_STORE.similarity_search_with_score(query, k=k)
        
        filtered_results = [doc for doc, score in results_with_scores if score >= self.RELEVANCE_THRESHOLD]

        # Fallback to top 2 raw results if no results meet threshold
        if len(filtered_results) == 0:
            logger.warning("No results met threshold, using top 2 raw results instead.")
            filtered_results = [doc for doc, _ in results_with_scores[:2]]
        return filtered_results
    
    def start_inference_session(self, 
                                query: str, 
                                history: str = "",
                                params: Optional[AIModelParameters] = AIModelParameters()) -> str:
        """
        Main method to handle the inference process.
        Args:
            query (str): The user's query.
            history (str): Conversation history for context.
            params (AIModelParameters): Inference parameters including temperature, max_tokens, etc.
        
        Returns:
            str: The response from the LLM based on the query and context.
        """
        filtered_results = self.perform_similarity_search(query=query)

        # Context construction
        if filtered_results and any(result.page_content.strip() for result in filtered_results):
            context = "\n\n".join([result.page_content for result in filtered_results])
            
            prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context, query=query, history=history)

        else:
            logger.warning("Context is empty after retrieval. Falling back to general knowledge answer.")
            prompt = f"Answer the following question based on your general knowledge:\n\nQuestion: {query}"

        try:
            response = invoke_model(prompt=prompt, parameters=params)
            return response

        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return "An error occurred while generating the response."
