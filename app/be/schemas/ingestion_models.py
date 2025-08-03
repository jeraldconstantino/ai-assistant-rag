from pydantic import BaseModel

class IngestionResponse(BaseModel):
    message: str

