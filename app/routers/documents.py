from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from typing import List
import uuid
from ..models.models import Document, DocumentResponse, Query, Answer, DocumentList, DocumentDetail
from ..databases.database import get_vector_store, VectorStore
from ..filters.embeddings import get_embeddings_model, chunk_text
import PyPDF2
import io
from openai import OpenAI

router = APIRouter()
client = OpenAI()

@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):

    if file.content_type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(await file.read()))
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text()
    elif file.content_type == "text/plain" or file.content_type == "text/markdown":
        content = (await file.read()).decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    document = Document(title=file.filename, content=content)
    vector_store = get_vector_store()
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(document.content)
    embeddings = get_embeddings_model(chunks)
    vector_store.add_document(doc_id, document.title, chunks, embeddings)

    return DocumentResponse(id=doc_id, title=document.title)


@router.post("/documents", response_model=DocumentResponse)
async def create_document(document: Document):
    vector_store = get_vector_store()
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(document.content)
    embeddings = get_embeddings_model(chunks)
    vector_store.add_document(doc_id, document.title, chunks, embeddings)
    
    return DocumentResponse(id=doc_id, title=document.title)

@router.post("/query", response_model=Answer)
async def query_documents(query: Query):
    vector_store = get_vector_store()

    query_embedding = get_embeddings_model([query.question])[0]

    relevant_chunks = vector_store.search(query_embedding, k=5)

    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    chunk_embeddings = [meta["embedding"] for meta in vector_store.chunk_metadata]
    
    from sentence_transformers import util

    sims = util.cos_sim(query_embedding, chunk_embeddings)[0].tolist()

    best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
    best_score = sims[best_idx]

    if best_score >= 0.65:  # threshold can be tuned
        best_chunk = vector_store.chunk_metadata[best_idx]
        sources = [DocumentResponse(id=best_chunk["doc_id"], title=best_chunk["title"])]
        return Answer(answer=best_chunk["chunk"], sources=sources)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use retrieved docs if useful."},
            {"role": "user", "content": query.question}
        ]
    )

    sources = [DocumentResponse(id=chunk["id"], title=chunk["title"]) for chunk in relevant_chunks]
    return Answer(answer=completion.choices[0].message.content, sources=sources)

@router.get("/documents", response_model=DocumentList)
async def list_documents():
    vector_store = get_vector_store()
    documents = vector_store.list_documents()
    return DocumentList(documents=[DocumentResponse(**doc) for doc in documents])

@router.get("/documents/{doc_id}", response_model=DocumentDetail)
async def get_document(doc_id: str):
    vector_store = get_vector_store()
    document = vector_store.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    vector_store = get_vector_store()
    if vector_store.delete_document(doc_id):
        return {"message": "Document deleted successfully"}
    raise HTTPException(status_code=404, detail="Document not found")
