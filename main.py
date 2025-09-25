from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import documents
from app.databases.database import initialize_vector_store, get_vector_store
from app.filters.embeddings import get_embeddings, chunk_text
import os

app = FastAPI(title="Document QA API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Initialize the vector store
    initialize_vector_store()
    
    # Load test data
    vector_store = get_vector_store()
    test_data_dir = "test_data"
    if os.path.exists(test_data_dir):
        for filename in os.listdir(test_data_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(test_data_dir, filename)
                with open(filepath, "r") as f:
                    content = f.read()
                
                doc_id = os.path.splitext(filename)[0]
                title = os.path.splitext(filename)[0].replace("_", " ").title()
                
                # Check if the document is already in the store
                if vector_store.get_document(doc_id) is None:
                    chunks = chunk_text(content)
                    embeddings = get_embeddings(chunks)
                    vector_store.add_document(doc_id, title, chunks, embeddings)

# Include routers
app.include_router(documents.router, prefix="/api", tags=["documents"])

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")
