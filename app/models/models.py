from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    title: str
    content: str

class DocumentResponse(BaseModel):
    id: str
    title: str

class DocumentDetail(DocumentResponse):
    content: str

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    sources: List[DocumentResponse]

class DocumentList(BaseModel):
    documents: List[DocumentResponse]