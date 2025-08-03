import shutil
from fastapi import APIRouter, File, UploadFile
from loguru import logger
from typing import List

# Custom libraries
from app.be.schemas.inference_models import (InferencePayload, 
                                             InferenceResponse, 
                                             AIModelParameters)
from app.be.schemas.ingestion_models import IngestionResponse
from app.be.utils.inference import ModelInference
from app.be.utils.ingestion import FileIngestor
from app.be.utils.model import invoke_model
from app.be.core.config import settings

router = APIRouter(prefix="/api", tags=["GenAI"])

@router.post("/inference", response_model=InferenceResponse)
def invoke_inference_session(items: InferencePayload):
    """Invoke the inference session with the provided query and parameters.
    Args:
        items (InferencePayload): The payload containing the query and parameters.

    Returns:
        InferenceResponse: The response from the inference session. 
    """
    logger.info(f"Starting inference session with query: {items.query}")
    inference = ModelInference()
    params = items.ai_model_parameters or AIModelParameters()
    response = inference.start_inference_session(query=items.query, 
                                                 history=items.history,
                                                 params=params)

    logger.info(f"Inference response: {response}")
    return InferenceResponse(response=response)


@router.post("/direct-inference", response_model=InferenceResponse)
def invoke_direct_inference_session(items: InferencePayload):
    """Invoke the inference session with the provided query and parameters.
    Args:
        items (InferencePayload): The payload containing the query and parameters.

    Returns:
        InferenceResponse: The response from the inference session. 
    """
    logger.info(f"Direct inference with query: {items.query}")
    params = items.ai_model_parameters or AIModelParameters()

    response = invoke_model(prompt=items.query, 
                            parameters=params)

    logger.info(f"Inference response: {response}")
    return InferenceResponse(response=response)


@router.post("/ingestion", response_model=IngestionResponse)
def invoke_ingestion_session(files: List[UploadFile] = File(...)):
    """Invoke the ingestion session with the provided query and parameters.
    Args:
        items (IngestionPayload): The payload containing the query and parameters.

    Returns:
        IngestionResponse: The response from the ingestion session. 
    """
    for file in files:
        file_path = f"{settings.src_data_path}/{file.filename}"
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            logger.info(f"File {file.filename} saved to {file_path}")
    
    ingestor = FileIngestor()
    ingestor.start_ingestion_session()

    # Return a success message
    logger.info("Files ingested successfully.")
    return IngestionResponse(message="Files ingested successfully.")
