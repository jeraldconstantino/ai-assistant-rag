import os
from pydantic_settings import BaseSettings

# Load the prompt template from a file
# This file contains the prompt template used for the AI model
prompt_template = ""
with open("app/be/core/prompt_template.txt", "r", encoding="utf-8") as file:
    prompt_template = file.read()
    
class Settings(BaseSettings):
    """Application configuration settings."""

    openai_api_key: str
    prompt_template: str = prompt_template

    # Paths
    vector_store_path: str = "app/be/data/vector_store"
    src_data_path: str = "app/be/data/raw"

    # Inference and ingestion settings
    embeddings_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-3.5-turbo"
    relevance_threshold: float = 0.6
    temperature: float = 0.0
    max_tokens: int = 750  
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def created_directories(path: str):
    """
    Create directories if they do not exist.
    
    Args:
        path (str): The directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

paths = [
    "app/be/data/raw",
    "app/be/data/vector_store"
]

for path in paths:
    created_directories(path)

settings = Settings()
