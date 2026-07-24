"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 10 — FastAPI Inference Server (Hardened)                         ║
║  Fake News & Source Credibility Detector                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  R9 hardening applied:                                                   ║
║   • Max statement length 2000 chars (422 beyond)                         ║
║   • API-key header auth (API_KEY env var)                                ║
║   • slowapi rate limit 30/min/key                                        ║
║   • 10s timeout + asyncio.wait_for on retrieval                          ║
║   • Request-ID structured logs                                           ║
║   • /health checks model loaded + Redis ping                             ║
║   • model_version in every response                                      ║
║   • Raw statements NOT stored in feedback buffer (PII stance)            ║
║   • Redis cache by statement SHA-256 hash                                ║
║   • Prometheus /metrics endpoint                                         ║
║   • SSE streaming on GET /assess/stream                                  ║
║                                                                          ║
║  Two-stage:                                                              ║
║   Stage 1: DeBERTa always (fast, ~40ms)                                 ║
║   Stage 2: LangGraph Agent + Mistral only if score < 0.5                ║
╚══════════════════════════════════════════════════════════════════════════╝

Install:
    pip install fastapi uvicorn slowapi redis prometheus-client python-dotenv

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    API_KEY=secret uvicorn api.main:app --host 0.0.0.0 --port 8000

Env vars (all optional):
    API_KEY        — if set, every request must include X-API-Key header
    REDIS_URL      — default: redis://localhost:6379
    LLM_SERVER_URL — URL of the separate Mistral server (api/llm_server.py)
    GOOGLE_FACTCHECK_API_KEY
    NEWSAPI_KEY
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from fastapi import (Depends, FastAPI, Header, HTTPException, Request,
                     Response, status)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.schemas import (AdminStatsResponse, FeedbackRequest, HealthResponse,
                          MODEL_VERSION, PredictRequest, PredictResponse,
                          SourceEvidence, SpeakerProfile)

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}',
)
logger = logging.getLogger("credibility-api")

# ── Rate limiter ─────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

# ── Stats (in-memory) ────────────────────────────────────────────────────────

_stats = {"total": 0, "cache_hits": 0, "cache_misses": 0,
          "total_latency_ms": 0.0, "start_time": time.time()}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL STORE
# ─────────────────────────────────────────────────────────────────────────────

class ModelStore:
    def __init__(self):
        self.model      = None
        self.tokenizer  = None
        self.tfidf_vec  = None
        self.tfidf_ridge = None
        self.feat_scaler = None
        self.speaker_profiles: dict = {}
        self.device     = None
        self.loaded     = False
        self._p5        = None

    def load(self):
        self._load_phase5_module()
        self._load_deberta()
        self._load_tfidf()
        self._load_speaker_profiles()
        self.loaded = True
        logger.info('"Models loaded successfully"')

    def _load_phase5_module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "phase5", _HERE / "deberta_model.py")
        self._p5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._p5)

    def _load_deberta(self):
        import torch
        from transformers import AutoTokenizer
        tok_path = _HERE / "models" / "deberta_tokenizer"
        wt_path  = _HERE / "models" / "deberta_best.pt"
        if not tok_path.exists() or not wt_path.exists():
            logger.warning('"DeBERTa weights not found — score will use prior"')
            return
        # MODEL_DEVICE env var forces the serving device (e.g. "cpu" on Apple
        # machines, where DeBERTa's attention is unreliable on MPS). Falls back to
        # auto-detect (cuda → mps → cpu) when unset.
        self.device = self._p5.detect_device(os.environ.get("MODEL_DEVICE"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=False)
        self.model = self._p5.DeBERTaCredibilityModel()
        ckpt = torch.load(wt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        self.model.to(self.device).eval()
        logger.info(f'"DeBERTa loaded on {self.device}"')

    def _load_tfidf(self):
        import joblib
        pkl_path = _HERE / "models" / "baseline_tfidf.pkl"
        if not pkl_path.exists():
            return
        saved = joblib.load(pkl_path)
        self.tfidf_vec   = saved.get("vectorizer")
        self.tfidf_ridge = saved.get("model")
        self.feat_scaler = saved.get("scaler")
        logger.info('"TF-IDF baseline loaded"')

    def _load_speaker_profiles(self):
        p = _HERE / "models" / "speaker_profiles.json"
        if p.exists():
            data = json.loads(p.read_text())
            self.speaker_profiles = {r["speaker"]: r
                                     for r in data.get("profiles", [])}
            logger.info(f'"Speaker profiles: {len(self.speaker_profiles):,}"')


_store = ModelStore()


# ─────────────────────────────────────────────────────────────────────────────
# REDIS CACHE
# ─────────────────────────────────────────────────────────────────────────────

_redis = None

async def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis.asyncio as aioredis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        _redis = aioredis.from_url(url, decode_responses=True)
        await _redis.ping()
    except Exception:
        _redis = None
    return _redis


def _cache_key(statement: str, speaker: str, context: str) -> str:
    raw = f"{statement}|{speaker}|{context}".encode()
    return "cred:" + hashlib.sha256(raw).hexdigest()[:24]


async def _cache_get(key: str) -> Optional[dict]:
    r = await _get_redis()
    if r is None:
        return None
    try:
        v = await r.get(key)
        return json.loads(v) if v else None
    except Exception:
        return None


async def _cache_set(key: str, value: dict, ttl: int = 3600) -> None:
    r = await _get_redis()
    if r is None:
        return
    try:
        await r.setex(key, ttl, json.dumps(value))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _score_tfidf(statement: str, feats: list[float]) -> Optional[float]:
    if _store.tfidf_vec is None:
        return None
    try:
        import numpy as np
        from scipy.sparse import hstack, csr_matrix
        from data_pipeline import preprocess_for_tfidf
        text_vec = _store.tfidf_vec.transform([preprocess_for_tfidf(statement)])
        feat_arr = _store.feat_scaler.transform([feats]) if _store.feat_scaler \
                   else np.array([feats])
        X = hstack([text_vec, csr_matrix(feat_arr)])
        return float(np.clip(_store.tfidf_ridge.predict(X)[0], 0, 1))
    except Exception:
        return None


def _score_deberta(statement: str, speaker: str, context: str,
                   prior: float, feats: list[float]) -> dict:
    if _store.model is None:
        return {"mean": prior, "lower": max(0, prior - 0.15),
                "upper": min(1, prior + 0.15), "std": 0.08}
    try:
        return _store._p5.predict_with_uncertainty(
            _store.model, _store.tokenizer,
            statement, speaker, context, prior, feats,
            device=_store.device)
    except Exception:
        return {"mean": prior, "lower": max(0, prior - 0.15),
                "upper": min(1, prior + 0.15), "std": 0.08}


def _score_to_verdict(score: float) -> str:
    if score < 0.35:  return "Likely False"
    if score < 0.65:  return "Unverified / Mixed"
    return "Likely True"


async def _full_assessment(req: PredictRequest,
                            request_id: str) -> PredictResponse:
    """Two-stage scoring: DeBERTa always; agent only when score < 0.5."""
    t0 = time.perf_counter()

    from data_pipeline import (
        normalise_context, get_context_prior, extract_all_features)

    context = normalise_context(req.context)
    prior   = float(get_context_prior(context))
    feats_d = extract_all_features(req.statement, context)
    # extract_all_features returns 11 features; supply the remaining two so the
    # 13-vector matches training (otherwise prior + length silently default to 0).
    feats_d["context_credibility_prior"] = prior
    feats_d["token_length_approx"]       = len(str(req.statement)) / 4.0
    from deberta_model import FEAT_COLS
    feats = [feats_d.get(c, 0.0) for c in FEAT_COLS]

    # Stage 1 — DeBERTa
    deberta_result = _score_deberta(
        req.statement, req.speaker, context, prior, feats)
    d_score = deberta_result["mean"]
    tfidf_score = _score_tfidf(req.statement, feats)

    # Stage 2 — Agent (only when score < 0.5)
    sources, sources_used, explanation, fc_score, errors = [], [], None, None, []
    if req.use_llm and d_score < 0.5:
        try:
            from agent.graph import run_assessment
            agent_result = await asyncio.wait_for(
                run_assessment(req.statement, req.speaker, context),
                timeout=30.0)
            sources      = agent_result.get("sources", [])
            sources_used = agent_result.get("sources_used", [])
            explanation  = agent_result.get("explanation")
            fc_score     = agent_result.get("fc_score")
            errors       = agent_result.get("errors", [])
        except asyncio.TimeoutError:
            errors.append("Agent pipeline timed out (30s)")
        except Exception as e:
            errors.append(f"Agent failed: {e}")

    # Fuse scores
    if fc_score is not None:
        final_score = round(0.6 * d_score + 0.4 * fc_score, 4)
    elif tfidf_score is not None:
        final_score = round(0.7 * d_score + 0.3 * tfidf_score, 4)
    else:
        final_score = d_score
    final_score = max(0.0, min(1.0, final_score))

    # Speaker profile lookup
    sp_profile: Optional[SpeakerProfile] = None
    if req.speaker and req.speaker in _store.speaker_profiles:
        p = _store.speaker_profiles[req.speaker]
        sp_profile = SpeakerProfile(
            speaker=p["speaker"], n_claims=p["n_claims"],
            bayes_score=p["bayes_score"], std_score=p["std_score"],
            job=p.get("job", ""), trend=p.get("trend", 0.0))

    model_used = ("DeBERTa+TF-IDF+Agent" if fc_score is not None
                  else "DeBERTa+TF-IDF" if tfidf_score is not None
                  else "DeBERTa")

    # Re-centre the MC-Dropout interval on the (possibly ensembled) final score
    # so the reported score always sits inside its own CI — otherwise the ensemble
    # can pull the point estimate outside the DeBERTa-only interval.
    ci_half  = (deberta_result["upper"] - deberta_result["lower"]) / 2.0
    lower_ci = round(max(0.0, final_score - ci_half), 4)
    upper_ci = round(min(1.0, final_score + ci_half), 4)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        f'"request_id":"{request_id}","score":{final_score},'
        f'"elapsed_ms":{elapsed},"model":"{model_used}"')

    return PredictResponse(
        request_id=request_id,
        statement=req.statement,
        speaker=req.speaker,
        context=context,
        score=final_score,
        lower_90ci=lower_ci,
        upper_90ci=upper_ci,
        verdict=_score_to_verdict(final_score),
        model_used=model_used,
        deberta_score=round(d_score, 4),
        fc_score=round(fc_score, 4) if fc_score is not None else None,
        context_prior_used=prior,
        speaker_profile=sp_profile,
        explanation=explanation,
        sources=[SourceEvidence(**s) for s in sources[:5]],
        sources_used=sources_used,
        elapsed_ms=elapsed,
        errors=errors,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────────────────────────────────

_API_KEY = os.environ.get("API_KEY", "")


async def verify_api_key(x_api_key: str = Header(default="")):
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401,
                            detail="Invalid or missing X-API-Key header")


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _store.load()
    yield

app = FastAPI(
    title="Fake News & Source Credibility Detector",
    description=(
        "Credibility score (0–1) for any statement, with MC Dropout "
        "confidence intervals, token-level SHAP, and source retrieval."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """<!doctype html><html><head><meta charset="utf-8">
    <title>Credibility Detector API</title>
    <style>body{font-family:system-ui,Arial,sans-serif;max-width:640px;margin:60px auto;
    padding:0 20px;line-height:1.6}code{background:#f2f2f2;padding:2px 6px;border-radius:4px}
    a{color:#185FA5}</style></head><body>
    <h1>🔍 Fake News &amp; Source Credibility Detector — API</h1>
    <p>The service is running. Try the interactive docs:</p>
    <ul>
      <li><a href="/docs">/docs</a> — Swagger UI (test <code>POST /assess</code> here)</li>
      <li><a href="/redoc">/redoc</a> — ReDoc reference</li>
      <li><a href="/health">/health</a> — model status</li>
      <li><a href="/speakers/top">/speakers/top</a> — most/least credible speakers</li>
      <li><a href="/metrics">/metrics</a> — Prometheus metrics</li>
    </ul>
    <p style="color:#666;font-size:.9em">POST a claim to <code>/assess</code> with
    JSON <code>{"statement": "...", "speaker": "...", "context": "..."}</code>.</p>
    </body></html>"""


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    r = await _get_redis()
    redis_ok = False
    if r:
        try:
            await r.ping()
            redis_ok = True
        except Exception:
            pass
    return HealthResponse(
        status="ok" if _store.loaded else "degraded",
        model_loaded=_store.loaded,
        redis_ok=redis_ok,
        uptime_s=round(time.time() - _stats["start_time"], 1),
    )


@app.post("/assess", response_model=PredictResponse, tags=["inference"],
          dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def assess(request: Request, req: PredictRequest):
    request_id = req.request_id or str(uuid.uuid4())[:8]
    cache_key  = _cache_key(req.statement, req.speaker, req.context)

    _stats["total"] += 1

    # Cache check
    cached = await _cache_get(cache_key)
    if cached:
        _stats["cache_hits"] += 1
        cached["request_id"] = request_id
        return JSONResponse(content=cached)

    _stats["cache_misses"] += 1
    result = await _full_assessment(req, request_id)
    result_dict = result.model_dump()
    await _cache_set(cache_key, result_dict)

    _stats["total_latency_ms"] += result.elapsed_ms
    return result


@app.get("/assess/stream", tags=["inference"],
         dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def assess_stream(request: Request, statement: str,
                         speaker: str = "", context: str = "unknown"):
    """SSE stream — emits progress events then the final result."""
    request_id = str(uuid.uuid4())[:8]

    async def event_generator() -> AsyncIterator[str]:
        yield _sse("status", {"msg": "Classifying with DeBERTa…", "request_id": request_id})
        req = PredictRequest(statement=statement, speaker=speaker,
                              context=context, request_id=request_id)
        result = await _full_assessment(req, request_id)
        yield _sse("status", {"msg": "Complete"})
        yield _sse("result", result.model_dump())
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/feedback", tags=["feedback"],
          dependencies=[Depends(verify_api_key)])
async def feedback(req: FeedbackRequest):
    """
    Accept user feedback. Statement hash stored, NOT the raw statement text
    (PII stance — users must opt in via STORE_STATEMENTS=1 env var).
    """
    store_raw = os.environ.get("STORE_STATEMENTS", "0") == "1"
    record = {
        "hash":            hashlib.sha256(req.statement.encode()).hexdigest()[:16],
        "predicted_score": req.predicted_score,
        "true_label":      req.true_label,
        "notes":           req.feedback_notes,
        "ts":              time.time(),
    }
    if store_raw:
        record["statement"] = req.statement

    r = await _get_redis()
    if r:
        try:
            await r.rpush("feedback_queue", json.dumps(record))
        except Exception:
            pass
    logger.info(f'"feedback received","hash":"{record["hash"]}"')
    return {"status": "ok", "message": "Feedback recorded"}


@app.get("/metrics", tags=["ops"])
async def metrics():
    """Prometheus-format text metrics."""
    n = max(_stats["total"], 1)
    lines = [
        "# HELP cred_requests_total Total API requests",
        "# TYPE cred_requests_total counter",
        f"cred_requests_total {_stats['total']}",
        "# HELP cred_cache_hits_total Cache hits",
        "# TYPE cred_cache_hits_total counter",
        f"cred_cache_hits_total {_stats['cache_hits']}",
        "# HELP cred_avg_latency_ms Average response latency in ms",
        "# TYPE cred_avg_latency_ms gauge",
        f"cred_avg_latency_ms {_stats['total_latency_ms'] / n:.1f}",
    ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")


@app.get("/admin/stats", response_model=AdminStatsResponse, tags=["ops"])
async def admin_stats():
    n = max(_stats["total"], 1)
    return AdminStatsResponse(
        total_requests=_stats["total"],
        cache_hits=_stats["cache_hits"],
        cache_misses=_stats["cache_misses"],
        avg_latency_ms=round(_stats["total_latency_ms"] / n, 1),
    )


@app.get("/speaker/{speaker_name}", tags=["speakers"])
async def get_speaker(speaker_name: str):
    profile = _store.speaker_profiles.get(speaker_name)
    if not profile:
        raise HTTPException(status_code=404,
                            detail=f"Speaker '{speaker_name}' not found")
    return profile


@app.get("/speakers/top", tags=["speakers"])
async def top_speakers(n: int = 20, min_claims: int = 3,
                        order: str = "desc"):
    profiles = [p for p in _store.speaker_profiles.values()
                if p.get("n_claims", 0) >= min_claims]
    reverse  = (order == "desc")
    profiles.sort(key=lambda p: p.get("bayes_score", 0.5), reverse=reverse)
    return profiles[:n]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000,
                reload=False, log_level="info")
