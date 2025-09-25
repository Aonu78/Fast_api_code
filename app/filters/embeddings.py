from openai import OpenAI
from typing import List
import os
from dotenv import load_dotenv
import re

load_dotenv()

client = OpenAI()

def get_embeddings_model(texts: List[str]) -> List[List[float]]:
    cleaned_texts = [clean_text(text) for text in texts]
    response = client.embeddings.create(
        input=cleaned_texts,
        model="text-embedding-ada-002"
    )
    return [embedding.embedding for embedding in response.data]

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    text = clean_text(text)
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_tokens = len(word) // 4
        if current_size + word_tokens > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_tokens
        else:
            current_chunk.append(word)
            current_size += word_tokens
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()
