"""
Phase 10 — Mistral LLM Microserver

Separate FastAPI process that loads the Mistral-7B adapter and handles
explanation generation requests. Decoupled from the main API so:
  • Main API stays fast when LLM is busy
  • LLM server can be on a separate GPU machine
  • Circuit breaker in main API degrades gracefully when LLM is down

Install:
    pip install fastapi uvicorn peft transformers bitsandbytes accelerate

Run (GPU required):
    uvicorn api.llm_server:app --host 0.0.0.0 --port 8001

Main API calls this via LLM_SERVER_URL env var.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ADAPTER_DIR = _HERE / "models" / "mistral_adapter"
_model      = None
_tokenizer  = None
_loaded     = False
_start_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ON STARTUP
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _loaded
    if ADAPTER_DIR.exists():
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base_name = "mistralai/Mistral-7B-Instruct-v0.2"
            base = AutoModelForCausalLM.from_pretrained(
                base_name, quantization_config=bnb,
                device_map="auto", trust_remote_code=False)
            _model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
            _tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
            _tokenizer.pad_token = _tokenizer.eos_token
            _model.eval()
            _loaded = True
            print(f"[LLM Server] Mistral adapter loaded from {ADAPTER_DIR}")
        except Exception as e:
            print(f"[LLM Server] WARNING: could not load adapter — {e}")
    else:
        print(f"[LLM Server] No adapter at {ADAPTER_DIR} — /generate will 422")
    yield


app = FastAPI(title="Mistral-7B Explanation Server",
              version="1.0.0", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    claim:          str   = Field(..., max_length=2000)
    speaker:        str   = Field(default="")
    context:        str   = Field(default="unknown")
    score:          float = Field(default=0.25, ge=0.0, le=1.0)
    label:          str   = Field(default="")
    max_new_tokens: int   = Field(default=256, ge=32, le=512)


class GenerateResponse(BaseModel):
    explanation: str
    elapsed_ms:  float
    model:       str = "mistral-7b-instruct-qlora"


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":     "ok" if _loaded else "no_model",
        "adapter":    str(ADAPTER_DIR) if _loaded else None,
        "uptime_s":   round(time.time() - _start_time, 1),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not _loaded:
        raise HTTPException(
            status_code=422,
            detail="Mistral adapter not loaded — run python llm_finetune.py --train first"
        )

    t0 = time.perf_counter()
    try:
        import torch
        from llm_finetune import build_prompt, _score_to_verdict

        verdict = _score_to_verdict(req.score)
        row = {
            "text":             req.claim,
            "speaker":          req.speaker,
            "context":          req.context,
            "credibility_score": req.score,
            "label_original":   req.label,
            "justification":    "",
        }
        prompt = build_prompt(row)
        instruction = prompt.split(" [/INST]")[0] + " [/INST]"

        inputs = _tokenizer(instruction, return_tensors="pt").to(_model.device)
        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=_tokenizer.eos_token_id,
            )
        explanation = _tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return GenerateResponse(
        explanation=explanation,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.llm_server:app", host="0.0.0.0", port=8001,
                reload=False, log_level="info")
