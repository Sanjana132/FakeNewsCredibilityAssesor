# Fake News & Source Credibility Detector

[![CI](https://github.com/Sanjana132/FakeNewsCredibilityAssesor/actions/workflows/ci.yml/badge.svg)](https://github.com/Sanjana132/FakeNewsCredibilityAssesor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end ML system that assigns any statement a **continuous credibility
score in [0, 1]** — not a binary real/fake label — together with a **calibrated
confidence interval**, a **token-level explanation** of what drove the verdict,
**retrieval of contradicting sources**, and an **LLM-generated justification**.

> Regression, not classification: "0.18 ± 0.06 — Likely False" is more honest and
> more useful than a hard "FAKE", and it lets the system express uncertainty.

---

## Highlights

- **~92k claims** merged and normalised from **4 public fact-checking datasets**
  (LIAR-2, MultiFC, FEVER, AVeriTeC) onto one 0–1 credibility scale.
- **Two-model stack**: a transparent **TF-IDF + Ridge** baseline (the MAE floor)
  and a fine-tuned **DeBERTa-v3** regressor with a fusion head over 13 engineered
  features.
- **Uncertainty you can trust**: **MC-Dropout** confidence intervals plus a
  **calibration** pass (reliability diagram, Expected Calibration Error,
  temperature scaling) — measured, not claimed.
- **Explainability**: token-level **SHAP** highlights (green = credibility-raising,
  red = credibility-lowering).
- **Agentic RAG**: a **LangGraph** pipeline that retrieves evidence from the Google
  Fact Check API, Wikipedia, a local FAISS index, and NewsAPI, then has a
  **Mistral-7B (QLoRA)** model write an evidence-grounded justification.
- **Production-minded**: hardened **FastAPI** service (API-key auth, rate limiting,
  Redis cache, SSE streaming), **Docker Compose** stack, **80 unit tests**, and
  **GitHub Actions CI**.
- **Reproducible**: global seeding, train-only data-driven priors (no leakage),
  and one-click **Google Colab GPU** notebooks.

---

## Results

Measured on the held-out test split (~8.9k claims). The fine-tuned DeBERTa-v3
regressor cuts test MAE **~13%** below the TF-IDF baseline.

| Model | Test MAE ↓ | Test Macro-F1 (3-class) ↑ |
|-------|-----------:|--------------------------:|
| TF-IDF + Ridge baseline (+ 13 features) | 0.2877 | 0.493 |
| **DeBERTa-v3-base (fusion head + MC-Dropout)** | **0.2512** | **0.555** |

Per-dataset test MAE (DeBERTa) — the gain is broad-based, strongest on the hard
political-claims slice, not carried by one easy corpus:

| Slice | LIAR-2 | MultiFC | AVeriTeC | FEVER |
|-------|-------:|--------:|---------:|------:|
| Test MAE ↓ | **0.203** | 0.234 | 0.266 | 0.281 |

*3-class buckets: false (<0.35), mixed (0.35–0.65), true (≥0.65). DeBERTa best
epoch 4, val MAE 0.2553. Reproduce with `python phase5_deberta.py --train
--device cuda --amp`; metrics are written to `models/deberta_results.json`.*

---

## Architecture

```
                 ┌─────────────────────────────────────────────┐
  raw claim ───► │ Phase 1–2  clean · normalise context (22     │
  + speaker      │            venues) · 13 engineered features  │
  + context      │            (VADER + opinion lexicon +        │
                 │            context×sentiment interactions)   │
                 └───────────────┬─────────────────────────────┘
                                 ▼
        ┌────────────────────────┴───────────────────────┐
        ▼                                                 ▼
 ┌───────────────┐                            ┌──────────────────────────┐
 │ TF-IDF+Ridge  │  baseline MAE floor        │ DeBERTa-v3 encoder        │
 │ (Phase 4)     │                            │ mean-pool → LayerNorm     │
 └───────────────┘                            │ ⊕ standardised features   │
                                              │ → fusion head → score     │
                                              │ + MC-Dropout 90% CI       │
                                              └────────────┬─────────────┘
                                                           ▼
        Phase 5b calibration (ECE, reliability, temperature scaling)
                                                           ▼
   if score < 0.5 ──► LangGraph agent (Phase 9): retrieve evidence
        (Google Fact Check · Wikipedia · FAISS · NewsAPI) ──► Mistral-7B
        (QLoRA, Phase 7) writes a grounded justification
                                                           ▼
   FastAPI service (Phase 8/10) · Gradio demo (Phase 12) · Docker (Phase 11)
```

---

## Datasets

| Source | Rows | What it adds |
|--------|-----:|--------------|
| [LIAR-2](https://huggingface.co/datasets/chengxuphd/liar2) | ~22.9k | PolitiFact political claims, 6-way labels, speaker credit history, justifications |
| [FEVER](https://huggingface.co/datasets/lucadiliello/fever) | ~47.8k (capped) | Wikipedia factual claims (supports/refutes/NEI) |
| [MultiFC](https://huggingface.co/datasets/pszemraj/multi_fc) | ~17.6k | 26 fact-checkers, health/science/social-media domains |
| [AVeriTeC](https://huggingface.co/datasets/pminervini/averitec) | ~3.5k | Web-verified claims with Q&A evidence chains |

All labels are mapped to a single **0.0–1.0** credibility scale, deduplicated,
and split 80/10/10 with **joint stratification** on credibility bucket × dataset.
Context priors are computed with **Bayesian shrinkage from the training split only**
— val/test labels never leak into features.

---

## Repository layout

| Phase | File(s) | Purpose |
|-------|---------|---------|
| 1–3 | `credibility_detector_phases123.py` | Load/merge datasets, feature engineering, EDA |
| 4 | `phase4_baseline_1.py` | TF-IDF + Ridge baseline + SHAP |
| 5 | `phase5_deberta.py` | DeBERTa-v3 fine-tuning, MC-Dropout CIs |
| 5b | `phase5b_calibration.py` | Reliability diagram, ECE, temperature scaling |
| 6 | `context_encoder.py`, `phase6_speaker_profiler.py` | Context embeddings, Bayesian speaker profiles |
| 7 | `llm_finetune.py`, `phase7_shap_explainer.py` | Mistral-7B QLoRA, token SHAP |
| 8 / 10 | `phase8_api.py`, `api/` | FastAPI inference (simple / hardened) |
| 9 | `agent/` | LangGraph agent + 4 retrieval tools |
| 11 | `Dockerfile.*`, `docker-compose.yml` | Containerised stack |
| 12 | `gradio_app.py` | Interactive demo UI |
| — | `config.py`, `utils/`, `tests/`, `.github/` | Config, seeding, 80 tests, CI |

---

## Quickstart

### Local (CPU is fine for phases 1–4; DeBERTa/Mistral want a GPU)

```bash
git clone https://github.com/Sanjana132/FakeNewsCredibilityAssesor.git
cd FakeNewsCredibilityAssesor
pip install -r requirements.api.txt          # inference/data stack
python -m nltk.downloader stopwords punkt punkt_tab opinion_lexicon

python credibility_detector_phases123.py     # Phases 1–3: build data + features + EDA
python phase4_baseline_1.py                   # Phase 4: TF-IDF baseline
uvicorn api.main:app --port 8000              # serve the API  → http://localhost:8000/docs
python gradio_app.py --standalone             # or the demo UI
```

### Train DeBERTa on a GPU (Google Colab)

Open **`colab_train_from_git.ipynb`** in Colab (`Runtime → T4 GPU → Run all`).
It clones this repo, installs the training stack, builds the data, and runs:

```bash
python phase5_deberta.py --train --device cuda
```

### Run the full stack with Docker

```bash
docker compose up redis api admin            # API + demo (+ llm service on a GPU host)
```

---

## Testing & CI

```bash
pip install -r requirements.dev.txt
pytest tests/ -v                              # 80 tests
```

Tests cover context normalisation, every dataset's label map, feature arithmetic,
the Bayesian-prior shrinkage (including the **train-only / no-leakage invariant**),
and a 200-row end-to-end pipeline smoke test. GitHub Actions runs them on every push.

---

## Engineering notes (things done deliberately right)

- **No data leakage** — context priors are fit on the train split only; a unit
  test asserts they're identical whether or not val/test rows exist.
- **Reproducibility** — `set_seed(42)` seeds Python/NumPy/PyTorch/langdetect at
  every entrypoint; dataset sampling is `shuffle(seed)`-then-select, never first-N.
- **Honest uncertainty** — calibration numbers (ECE, CI coverage) are measured and
  saved to `models/calibration.json`; nothing is claimed without them.
- **Feature/serving parity** — the API computes inference features via the exact
  training function, so production scores match evaluation.

## Limitations & roadmap

- Labels come from fact-checkers and inherit their topical and temporal biases;
  the score reflects "how fact-checkers would rate this", not absolute ground truth.
- The Mistral justification layer and FAISS evidence index are optional and need a
  GPU / scraped index respectively.
- **Roadmap (not implemented):** continuous learning from user feedback, drift
  monitoring (Evidently), and a managed cloud deployment.

---

## License

MIT — see `LICENSE`.

*Built as an end-to-end ML engineering portfolio project: data pipeline →
classical baseline → transformer fine-tuning → calibration → LLM + RAG agent →
API → containerisation → tests/CI.*
