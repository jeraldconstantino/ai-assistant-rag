import os
from pydantic_settings import BaseSettings
    
class Settings(BaseSettings):
    """Application configuration settings."""

    # UI Settings
    title: str = "RAG AI Assistant"
    height: int = 500
    icon: str = ":robot_face:"
    layout: str = "wide"

    # API Endpoints
    base_url: str = "http://localhost:8000"
    inference_endpoint: str = f"{base_url}/api/inference"
    direct_inference_endpoint: str = f"{base_url}/api/direct-inference"
    ingestion_endpoint: str = f"{base_url}/api/ingestion"

    # class Config:
    #     env_file = ".env"
    #     env_file_encoding = "utf-8"

# def created_directories(path: str):
#     """
#     Create directories if they do not exist.
    
#     Args:
#         path (str): The directory path to create.
#     """
#     if not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)

# paths = [
#     "app/be/data/raw",
#     "app/be/data/vector_store"
# ]

# for path in paths:
#     created_directories(path)

settings = Settings()
