"""
Phase 10 — Pydantic schemas for the FastAPI server.

All request and response models live here so api/main.py
and tests can import them without circular dependencies.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


MODEL_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    statement: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The claim or statement to assess (10–2000 characters)",
        examples=["The unemployment rate fell to a record low last month."],
    )
    speaker: str = Field(
        default="",
        max_length=200,
        description="Person or entity making the claim",
    )
    context: str = Field(
        default="unknown",
        max_length=300,
        description="Venue or medium (e.g. 'a campaign rally', 'a TV interview')",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-supplied request ID for correlation",
    )
    use_llm: bool = Field(
        default=True,
        description="Whether to call the LLM for explanation when score < 0.5",
    )

    @field_validator("statement")
    @classmethod
    def statement_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("statement must not be blank")
        return v.strip()


class FeedbackRequest(BaseModel):
    statement:       str   = Field(..., min_length=10, max_length=2000)
    predicted_score: float = Field(..., ge=0.0, le=1.0)
    true_label:      str   = Field(...,
                                    description="true | false | mixed | unknown")
    feedback_notes:  str   = Field(default="", max_length=1000)

    @field_validator("true_label")
    @classmethod
    def valid_label(cls, v: str) -> str:
        allowed = {"true", "false", "mixed", "unknown"}
        if v.lower() not in allowed:
            raise ValueError(f"true_label must be one of {allowed}")
        return v.lower()


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class SourceEvidence(BaseModel):
    title:     str
    snippet:   str
    url:       str
    score:     Optional[float]
    source:    str   # "google_fc" | "wikipedia" | "faiss" | "newsapi"
    relevance: float


class SpeakerProfile(BaseModel):
    speaker:     str
    n_claims:    int
    bayes_score: float
    std_score:   float
    job:         str
    trend:       float


class PredictResponse(BaseModel):
    request_id:           str
    statement:            str
    speaker:              str
    context:              str
    score:                float = Field(..., ge=0.0, le=1.0,
                                        description="Point credibility estimate")
    lower_90ci:           float = Field(..., ge=0.0, le=1.0)
    upper_90ci:           float = Field(..., ge=0.0, le=1.0)
    verdict:              str   = Field(...,
                                        description="Likely True | Unverified / Mixed | Likely False")
    model_used:           str
    model_version:        str   = MODEL_VERSION
    deberta_score:        Optional[float] = None
    fc_score:             Optional[float] = None
    context_prior_used:   float
    speaker_profile:      Optional[SpeakerProfile] = None
    explanation:          Optional[str] = None
    sources:              list[SourceEvidence] = Field(default_factory=list)
    sources_used:         list[str] = Field(default_factory=list)
    elapsed_ms:           float
    errors:               list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status:        str            # "ok" | "degraded"
    model_loaded:  bool
    redis_ok:      bool
    model_version: str = MODEL_VERSION
    uptime_s:      float


class AdminStatsResponse(BaseModel):
    total_requests:   int
    cache_hits:       int
    cache_misses:     int
    avg_latency_ms:   float
    model_version:    str = MODEL_VERSION
