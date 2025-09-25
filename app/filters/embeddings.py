from openai import OpenAI
from typing import List
import os
from dotenv import load_dotenv
import re

load_dotenv()

client = OpenAI()

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using OpenAI's ada-002 model."""
    # Clean texts before getting embeddings
    cleaned_texts = [clean_text(text) for text in texts]
    response = client.embeddings.create(
        input=cleaned_texts,
        model="text-embedding-ada-002"
    )
    return [embedding.embedding for embedding in response.data]

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of approximately equal size."""
    # Clean the text first
    text = clean_text(text)
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        # Use estimate_token_count to check chunk size
        word_tokens = estimate_token_count(word)
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
    """Clean text by removing extra whitespace and special characters."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def validate_document_content(content: str) -> bool:
    """Validate document content."""
    if not content or len(content.strip()) == 0:
        return False
    return True

def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a unique ID for document chunks."""
    return f"{doc_id}_{chunk_index}"

def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in a text (rough approximation)."""
    # Rough estimation: 1 token ≈ 4 characters
    return len(text) // 4
