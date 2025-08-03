from fastapi import FastAPI
from app.be.api import routes

app = FastAPI(title="RAG AI Assistant App", version="1.0")

app.include_router(routes.router)

@app.get("/")
def root():
    return {"status": "200 OK"}