from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from typing import List
import uuid
from ..models.models import Document, DocumentResponse, Query, Answer, DocumentList, DocumentDetail
from ..databases.database import get_vector_store, VectorStore
from ..filters.embeddings import get_embeddings, chunk_text
import openai
import PyPDF2
import io

router = APIRouter()

@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    # Extract text from file
    if file.content_type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(await file.read()))
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text()
    elif file.content_type == "text/plain" or file.content_type == "text/markdown":
        content = (await file.read()).decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Create document
    document = Document(title=file.filename, content=content)

    # Use existing create_document logic
    vector_store = get_vector_store()

    # Generate document ID
    doc_id = str(uuid.uuid4())

    # Split document into chunks
    chunks = chunk_text(document.content)

    # Get embeddings for chunks
    embeddings = get_embeddings(chunks)

    # Store document in vector store
    vector_store.add_document(doc_id, document.title, chunks, embeddings)

    return DocumentResponse(id=doc_id, title=document.title)


@router.post("/documents", response_model=DocumentResponse)
async def create_document(document: Document):
    vector_store = get_vector_store()
    
    # Generate document ID
    doc_id = str(uuid.uuid4())
    
    # Split document into chunks
    chunks = chunk_text(document.content)
    
    # Get embeddings for chunks
    embeddings = get_embeddings(chunks)
    
    # Store document in vector store
    vector_store.add_document(doc_id, document.title, chunks, embeddings)
    
    return DocumentResponse(id=doc_id, title=document.title)

@router.post("/query", response_model=Answer)
async def query_documents(query: Query):
    vector_store = get_vector_store()
    
    # Get query embedding
    query_embedding = get_embeddings([query.question])[0]
    
    # Search for relevant chunks
    relevant_chunks = vector_store.search(query_embedding)
    
    if not relevant_chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")
    
    # Prepare context for GPT
    context = "\n".join([chunk["chunk"] for chunk in relevant_chunks])
    
    # Generate answer using GPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query.question}"}
        ]
    )
    
    answer = response.choices[0].message.content
    
    # Prepare source documents
    sources = [DocumentResponse(id=chunk["id"], title=chunk["title"]) 
              for chunk in relevant_chunks]
    
    return Answer(answer=answer, sources=sources)

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