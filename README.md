Project Architecture 

Streamlit UI ->FAST API ENDPOINT -> RAG Agent -> Hybrid Retriever/OPEN AI LLM - > Postgres -> Chunking + Embeddings

Phase 1 - Fast API
Phase 2 - Database connection
Phase 3 - Core
Phase 4 - Upload API
Phase 5 - PDF Integration
Phase 6 - Pgvector, Chunking, embeddings 
Phase 7 - RAG Agnet Creation
Phase 8 - Streamlit creation
Phase 9 - End to End Validatin with Swagger/Postman



FastAPI as the backend
Streamlit as the frontend
LangChain for orchestration
PostgreSQL + pgvector for storage
Hybrid Retrieval (Vector + FTS)
Agent-based architecture
Logging
Configuration management
Unit tests
Docker support
GitHub Actions (CI/CD)
#comments
uv run uvicorn app.main:app --reload
