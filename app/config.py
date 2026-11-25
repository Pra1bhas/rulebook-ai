"""All settings in one place.

Values are read from environment variables (or a .env file), so the same code
runs on your laptop, inside Docker and on the deployed server without edits.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (free tier) ---
    llm_provider: str = "gemini"  # "gemini" | "groq" | "fake" (fake = tests only)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_max_retries: int = 4

    # --- Embeddings + reranker (run locally, free) ---
    embedder: str = "fastembed"  # "fastembed" | "fake" (fake = tests only)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker: str = "fastembed"  # "fastembed" | "none"
    reranker_model: str = "BAAI/bge-reranker-base"
    model_cache_dir: str = "models"

    # --- Vector DB ---
    # If qdrant_url is set we talk to a Qdrant server (docker-compose).
    # Otherwise Qdrant runs inside our own process and saves to qdrant_path.
    qdrant_url: str = ""
    qdrant_path: str = "storage/qdrant"
    collection: str = "rulebook"

    # --- Chunking ---
    chunk_size_words: int = 220
    chunk_overlap_words: int = 40

    # --- Retrieval ---
    candidates_k: int = 20  # how many results each search returns before merging
    final_k: int = 5  # how many chunks go to the LLM
    rrf_k: int = 60  # constant in the Reciprocal Rank Fusion formula
    rewrite_query: bool = True  # turn Tenglish questions into search-friendly English

    # --- Problem solver ---
    solve_k: int = 8  # chunks sent to the LLM for an action plan
    solve_per_query_k: int = 4  # best chunks kept from each sub-query

    # --- User uploads (private per browser session) ---
    max_user_docs: int = 5
    max_upload_mb: int = 20
    max_upload_pages: int = 300
    user_doc_ttl_hours: float = 24  # uploads are auto-deleted after this

    # --- Protecting the free tier ---
    rate_per_minute: int = 6  # LLM questions per user per minute
    rate_per_day: int = 60
    cache_size: int = 500  # same question again -> saved answer, no LLM call
    cache_ttl_hours: float = 24
    admin_key: str = ""  # set it to allow POST /admin/ingest (adds PUBLIC documents)

    # --- Paths ---
    data_dir: str = "data/raw"
    storage_dir: str = "storage"
    auto_ingest: bool = True  # on startup, index data/raw if the index is empty

    @property
    def chunks_file(self) -> Path:
        return Path(self.storage_dir) / "chunks.jsonl"


settings = Settings()
