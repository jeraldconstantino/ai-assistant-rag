import requests
from core.config import settings

def ingest_files(files):
    response = requests.post(settings.ingestion_endpoint, files=files)
    return response