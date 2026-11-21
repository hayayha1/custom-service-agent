"""Application configuration backed by Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (OpenAI-compatible)
    llm_api_key: str = Field(default="", description="API key for primary chat LLM")
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI-compatible LLM endpoint",
    )
    llm_model: str = Field(default="gpt-4o-mini", description="Primary chat model name")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    memory_llm_model: str = Field(
        default="",
        description="Optional lighter model for memory extraction; falls back to llm_model",
    )

    # Embeddings
    embedding_api_key: str = Field(
        default="",
        description="API key for embedding model; falls back to llm_api_key if empty",
    )
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for embedding endpoint",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    embedding_dim: int = Field(default=1536, gt=0)

    # Reranker (OpenAI/Cohere/Jina/SiliconFlow/DashScope compatible /rerank endpoint)
    reranker_enabled: bool = Field(default=False, description="Enable Cross-Encoder Reranker")
    reranker_api_key: str = Field(default="", description="API key for reranker service")
    reranker_base_url: str = Field(
        default="https://api.siliconflow.cn/v1/rerank",
        description="HTTP endpoint for reranker service",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Reranker model name",
    )
    reranker_top_n: int = Field(default=4, ge=1, le=20)
    reranker_min_score: float = Field(default=0.25, ge=0.0, le=1.0)

    # Phase 2: Knowledge Base Cleaning & Chunking
    kb_chunk_size: int = Field(default=450, ge=100, le=4000)
    kb_chunk_overlap: int = Field(default=70, ge=0, le=1000)
    kb_redact_pii: bool = Field(default=True)

    # Phase 2: Hybrid Retrieval & RRF Fusion
    retrieval_rewrite_enabled: bool = Field(default=True)
    retrieval_dense_top_k: int = Field(default=15, ge=1, le=50)
    retrieval_sparse_top_k: int = Field(default=15, ge=1, le=50)
    retrieval_rrf_k: int = Field(default=60, ge=1, le=200)
    retrieval_rrf_top_k: int = Field(default=10, ge=1, le=50)
    retrieval_dense_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    retrieval_sparse_weight: float = Field(default=1.0, ge=0.0, le=10.0)

    # Phase 2: Langfuse Observability
    langfuse_enabled: bool = Field(default=False)
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # Runtime persisted model settings override file
    model_settings_path: str = Field(
        default="data/model_settings.json",
        description="Path to persisted runtime model configuration overrides",
    )

    # PostgreSQL (pgvector + LangGraph Checkpointer)
    postgres_dsn: str = Field(
        default="postgresql://cs_agent:cs_agent_pass@127.0.0.1:5433/cs_agent?sslmode=disable",
        description="PostgreSQL connection string",
    )
    postgres_pool_min_size: int = Field(default=1, ge=1)
    postgres_pool_max_size: int = Field(default=10, ge=1)
    local_dev_fallback: bool = Field(
        default=True,
        description="Fallback to in-memory stores & fakeredis when local Docker Postgres/Redis is not running",
    )

    # Redis (Short-term memory + Session Lock + Tool Cache)
    redis_url: str = Field(default="redis://127.0.0.1:6379/0")

    # Agent Loop & Guardrails
    max_tool_iterations: int = Field(default=5, ge=1, le=20)
    tool_timeout_seconds: float = Field(default=10.0, gt=0.0)
    tool_cache_ttl_seconds: int = Field(default=60, ge=0)

    # Memory Tuning
    short_term_max_turns: int = Field(default=10, ge=1)
    session_ttl_seconds: int = Field(default=1800, ge=60)
    session_lock_ttl_seconds: int = Field(default=120, ge=5)
    session_lock_wait_seconds: float = Field(default=5.0, ge=0.0)
    long_term_top_k: int = Field(default=3, ge=1, le=20)
    long_term_min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    long_term_summary_every_n_turns: int = Field(default=10, ge=1)

    @property
    def effective_memory_model(self) -> str:
        return self.memory_llm_model or self.llm_model

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
