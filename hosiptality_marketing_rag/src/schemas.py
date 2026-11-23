"""Pydantic models shared across the pipeline."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

DocType = Literal["case_study", "campaign_report", "brand_guidelines"]


class ChunkMetadata(BaseModel):
    source: str
    section: str
    page: int = Field(ge=0)
    property: str
    period: str
    channel: str
    segment: str
    doc_type: DocType

    @field_validator("source", "section", "property", "period", "channel", "segment")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("metadata fields must be non-empty (use 'all'/'unknown')")
        return v


class Chunk(BaseModel):
    chunk_id: str
    text: str                      # text used for retrieval (table md + linearized rows for tables)
    kind: Literal["text", "table"] = "text"
    table_md: Optional[str] = None  # full untouched markdown table (table chunks only)
    metadata: ChunkMetadata


class RewrittenQuery(BaseModel):
    original: str
    search_queries: list[str] = Field(min_length=1, max_length=3)
    filters: dict[str, str] = Field(default_factory=dict)
    expansions: dict[str, str] = Field(default_factory=dict)  # acronym -> expansion applied

    @field_validator("search_queries")
    @classmethod
    def _cap(cls, v: list[str]) -> list[str]:
        return v[:3]


class Claim(BaseModel):
    text: str
    value: str
    supported: bool
    evidence_chunk: Optional[str] = None


class VerificationResult(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    unsupported_claims: list[Claim] = Field(default_factory=list)
    citation_errors: list[str] = Field(default_factory=list)
    judge_unsupported: list[str] = Field(default_factory=list)
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    is_refusal: bool = False


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    score: float = 0.0
    kind: str = "text"


class QueryResponse(BaseModel):
    question: str
    rewritten: RewrittenQuery
    answer: str
    citations: list[str]
    contexts: list[RetrievedChunk]
    verification: VerificationResult
    latency_ms: dict[str, float]
