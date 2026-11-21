"""Runtime model settings persistence and masked view for LLM, Embedding, and Reranker."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.config import Settings

logger = logging.getLogger(__name__)


def mask_secret(secret: str) -> str:
    """Mask an API key for safe display in UI responses."""
    cleaned = secret.strip()
    if not cleaned:
        return ""
    if len(cleaned) <= 8:
        return "*" * len(cleaned)
    return f"{cleaned[:4]}...{cleaned[-4:]}"


class LLMConfigView(BaseModel):
    base_url: str
    model: str
    temperature: float
    memory_model: str = ""
    api_key_masked: str = ""
    has_api_key: bool = False


class EmbeddingConfigView(BaseModel):
    base_url: str
    model: str
    dim: int
    api_key_masked: str = ""
    has_api_key: bool = False


class RerankerConfigView(BaseModel):
    enabled: bool
    base_url: str
    model: str
    top_n: int
    min_score: float
    api_key_masked: str = ""
    has_api_key: bool = False


class LangfuseConfigView(BaseModel):
    enabled: bool = False
    host: str = "https://cloud.langfuse.com"
    public_key_masked: str = ""
    secret_key_masked: str = ""
    has_keys: bool = False


class ModelSettingsResponse(BaseModel):
    llm: LLMConfigView
    embedding: EmbeddingConfigView
    reranker: RerankerConfigView
    langfuse: LangfuseConfigView = Field(default_factory=LangfuseConfigView)


class LLMConfigUpdate(BaseModel):
    base_url: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    memory_model: str = ""
    api_key: str | None = Field(
        default=None,
        description="Leave None or empty string to keep existing API key",
    )


class EmbeddingConfigUpdate(BaseModel):
    base_url: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    dim: int = Field(default=1536, gt=0)
    api_key: str | None = Field(
        default=None,
        description="Leave None or empty string to keep existing API key",
    )


class RerankerConfigUpdate(BaseModel):
    enabled: bool = False
    base_url: str = Field(default="https://api.siliconflow.cn/v1/rerank")
    model: str = Field(default="BAAI/bge-reranker-v2-m3")
    top_n: int = Field(default=4, ge=1, le=20)
    min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    api_key: str | None = Field(
        default=None,
        description="Leave None or empty string to keep existing API key",
    )


class LangfuseConfigUpdate(BaseModel):
    enabled: bool = False
    host: str = Field(default="https://cloud.langfuse.com")
    public_key: str | None = None
    secret_key: str | None = None


class ModelSettingsUpdateRequest(BaseModel):
    llm: LLMConfigUpdate
    embedding: EmbeddingConfigUpdate
    reranker: RerankerConfigUpdate
    langfuse: LangfuseConfigUpdate | None = None


class ModelSettingsStore:
    """Manages runtime LLM / Embedding / Reranker settings with JSON file persistence."""

    def __init__(self, settings: Settings, storage_path: str | Path | None = None) -> None:
        self._path = Path(storage_path or settings.model_settings_path)
        self._data: dict[str, Any] = {
            "llm": {
                "base_url": settings.llm_base_url,
                "model": settings.llm_model,
                "temperature": settings.llm_temperature,
                "memory_model": settings.memory_llm_model,
                "api_key": settings.llm_api_key,
            },
            "embedding": {
                "base_url": settings.embedding_base_url,
                "model": settings.embedding_model,
                "dim": settings.embedding_dim,
                "api_key": settings.embedding_api_key,
            },
            "reranker": {
                "enabled": settings.reranker_enabled,
                "base_url": settings.reranker_base_url,
                "model": settings.reranker_model,
                "top_n": settings.reranker_top_n,
                "min_score": settings.reranker_min_score,
                "api_key": settings.reranker_api_key,
            },
            "langfuse": {
                "enabled": settings.langfuse_enabled,
                "host": settings.langfuse_host,
                "public_key": settings.langfuse_public_key,
                "secret_key": settings.langfuse_secret_key,
            },
        }
        self._load_overrides()

    def _load_overrides(self) -> None:
        if not self._path.exists():
            return
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for section in ("llm", "embedding", "reranker", "langfuse"):
                    if isinstance(saved.get(section), dict):
                        self._data[section].update(saved[section])
        except Exception as exc:
            logger.warning("Failed to load model settings overrides from %s: %s", self._path, exc)

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_raw(self) -> dict[str, Any]:
        return {k: dict(v) for k, v in self._data.items()}

    def get_masked(self) -> ModelSettingsResponse:
        llm = self._data["llm"]
        emb = self._data["embedding"]
        rr = self._data["reranker"]
        lf = self._data["langfuse"]
        return ModelSettingsResponse(
            llm=LLMConfigView(
                base_url=llm["base_url"],
                model=llm["model"],
                temperature=float(llm["temperature"]),
                memory_model=llm.get("memory_model", ""),
                api_key_masked=mask_secret(llm.get("api_key", "")),
                has_api_key=bool(llm.get("api_key", "").strip()),
            ),
            embedding=EmbeddingConfigView(
                base_url=emb["base_url"],
                model=emb["model"],
                dim=int(emb["dim"]),
                api_key_masked=mask_secret(emb.get("api_key", "")),
                has_api_key=bool(emb.get("api_key", "").strip()),
            ),
            reranker=RerankerConfigView(
                enabled=bool(rr["enabled"]),
                base_url=rr["base_url"],
                model=rr["model"],
                top_n=int(rr["top_n"]),
                min_score=float(rr["min_score"]),
                api_key_masked=mask_secret(rr.get("api_key", "")),
                has_api_key=bool(rr.get("api_key", "").strip()),
            ),
            langfuse=LangfuseConfigView(
                enabled=bool(lf.get("enabled", False)),
                host=str(lf.get("host") or "https://cloud.langfuse.com"),
                public_key_masked=mask_secret(str(lf.get("public_key") or "")),
                secret_key_masked=mask_secret(str(lf.get("secret_key") or "")),
                has_keys=bool(
                    str(lf.get("public_key") or "").strip()
                    and str(lf.get("secret_key") or "").strip()
                ),
            ),
        )

    def update(self, payload: ModelSettingsUpdateRequest) -> ModelSettingsResponse:
        self._data["llm"]["base_url"] = payload.llm.base_url.strip()
        self._data["llm"]["model"] = payload.llm.model.strip()
        self._data["llm"]["temperature"] = payload.llm.temperature
        self._data["llm"]["memory_model"] = payload.llm.memory_model.strip()
        if payload.llm.api_key and payload.llm.api_key.strip():
            self._data["llm"]["api_key"] = payload.llm.api_key.strip()

        self._data["embedding"]["base_url"] = payload.embedding.base_url.strip()
        self._data["embedding"]["model"] = payload.embedding.model.strip()
        self._data["embedding"]["dim"] = payload.embedding.dim
        if payload.embedding.api_key and payload.embedding.api_key.strip():
            self._data["embedding"]["api_key"] = payload.embedding.api_key.strip()

        self._data["reranker"]["enabled"] = payload.reranker.enabled
        self._data["reranker"]["base_url"] = payload.reranker.base_url.strip()
        self._data["reranker"]["model"] = payload.reranker.model.strip()
        self._data["reranker"]["top_n"] = payload.reranker.top_n
        self._data["reranker"]["min_score"] = payload.reranker.min_score
        if payload.reranker.api_key and payload.reranker.api_key.strip():
            self._data["reranker"]["api_key"] = payload.reranker.api_key.strip()

        if payload.langfuse is not None:
            self._data["langfuse"]["enabled"] = payload.langfuse.enabled
            self._data["langfuse"]["host"] = payload.langfuse.host.strip() or "https://cloud.langfuse.com"
            if payload.langfuse.public_key and payload.langfuse.public_key.strip():
                self._data["langfuse"]["public_key"] = payload.langfuse.public_key.strip()
            if payload.langfuse.secret_key and payload.langfuse.secret_key.strip():
                self._data["langfuse"]["secret_key"] = payload.langfuse.secret_key.strip()

        self._persist()
        return self.get_masked()
