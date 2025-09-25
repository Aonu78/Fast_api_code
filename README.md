# Document QA API

This is a FastAPI-based document question-answering system that uses vector embeddings and LLMs to provide accurate answers based on ingested documents.

## Features

- Document ingestion and indexing
- Semantic search using vector embeddings (OpenAI ada-002)
- Answer generation using GPT-3.5/4
- Document management (list, delete)
- FAISS vector store for efficient similarity search

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /api/documents
Upload a new document for indexing.

Request body:
```json
{
    "title": "Document Title",
    "content": "Document content goes here..."
}
```

### POST /api/query
Ask a question about the documents.

Request body:
```json
{
    "question": "Your question here?"
}
```

### GET /api/documents
List all indexed documents.

### DELETE /api/documents/{doc_id}
Delete a specific document by ID.

## Libraries Used

- FastAPI: Modern, fast web framework for building APIs
- FAISS: Efficient similarity search and clustering of dense vectors
- OpenAI: For generating embeddings and answers
- LangChain: For document processing and LLM integration
- Python-multipart: For handling file uploads
- Pydantic: For data validation
- uvicorn: ASGI server

## Testing

To test the API endpoints:

1. Start the server
2. Visit `http://localhost:8000/docs` for the Swagger UI
3. Try out the endpoints with the interactive documentation

Sample test data is provided in the `test_data` directory.