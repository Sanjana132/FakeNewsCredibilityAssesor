# Running on Google Colab (GPU)

Everything needed to train and evaluate the **Fake News & Source Credibility
Detector** on a Colab GPU. The heavy work — DeBERTa fine-tuning (Phase 5) and
the optional Mistral-7B QLoRA (Phase 7) — needs CUDA, which Colab provides free
(T4) or paid (A100/L4).

---

## What's in this folder

| File / dir | Purpose |
|------------|---------|
| `run_on_colab.ipynb` | **The notebook to open in Colab.** Runs every phase in order. |
| `credibility_detector_phases123.py` | Phases 1–3: data loading, features, EDA |
| `phase4_baseline_1.py` | Phase 4: TF-IDF + Ridge baseline (MAE benchmark) |
| `phase5_deberta.py` | Phase 5: DeBERTa-v3 fine-tuning (**GPU**) |
| `phase5b_calibration.py` | Phase 5b: reliability diagram, ECE, CI coverage |
| `context_encoder.py` | Phase 6: learned context-slot embeddings |
| `phase6_speaker_profiler.py` | Phase 6: Bayesian speaker credibility profiles |
| `phase7_shap_explainer.py` | Phase 7: token-level SHAP explanations |
| `llm_finetune.py` | Phase 7: Mistral-7B QLoRA justification model (**A100**) |
| `phase8_api.py`, `api/`, `agent/` | Serving + retrieval agent (run locally, not on Colab) |
| `speaker_scraper.py`, `gradio_app.py` | Scraper + demo UI |
| `config.py`, `utils/` | Shared config + reproducibility seeding |
| `requirements.*.txt` | Dependency sets (`train` pulls in `api`) |

---

## Fastest path (recommended): clone from GitHub

You don't actually need to upload this folder — the notebook clones the repo.

1. Open **[colab.research.google.com](https://colab.research.google.com)**.
2. **File → Upload notebook** → pick `run_on_colab.ipynb` from this folder.
3. **Runtime → Change runtime type → T4 GPU** (or A100 if you'll run the LLM step). Save.
4. Run the cells **top to bottom** (`Shift+Enter`, or *Runtime → Run all*).
   - Cell 2 clones `https://github.com/Sanjana132/FakeNewsCredibilityAssesor`.
   - Cell 3 installs dependencies (~2–3 min).
   - Cell 6 is the main training step (~40–60 min on T4).
5. Cell 11 saves the trained weights to your Google Drive.

## Offline path: upload this folder as a zip

If you'd rather not clone (e.g. private fork, no internet in the notebook):

1. On your machine, zip this folder:
   ```bash
   cd /Users/sanjanareddy/FakeNews_ML
   zip -r colab_bundle.zip colab_bundle
   ```
2. Open `run_on_colab.ipynb` in Colab, set the **T4 GPU** runtime.
3. **Skip Cell 2 (clone).** Instead, run **Cell "Option B"**: it uploads
   `colab_bundle.zip`, unzips it, and `cd`s into the folder.
4. Continue from Cell 3 (install) onward.

---

## Step-by-step (what each notebook cell does)

| Cell | Phase | Command | Time (T4) |
|------|-------|---------|-----------|
| 1 | — | Verify GPU is attached | instant |
| 2 | — | `git clone` the repo | ~10 s |
| 3 | — | `pip install -r requirements.train.txt` + NLTK data | ~2–3 min |
| 4 | 1–3 | `python credibility_detector_phases123.py` (builds `data/*.csv`) | ~8–12 min |
| 5 | 4 | `python phase4_baseline_1.py --no-shap` | ~2 min |
| 6 | 5 | `python phase5_deberta.py --train --device cuda` ⭐ | **~40–60 min** |
| 7 | 5b | `python phase5b_calibration.py --device cuda` | ~3 min |
| 8 | 6 | `phase6_speaker_profiler.py` + `context_encoder.py --visualise` | ~1 min |
| 9 | 7 | `python phase7_shap_explainer.py` | ~3–5 min |
| 10 | 7 | `python llm_finetune.py --train` (**A100 only**, optional) | ~1–2 hr |
| 11 | — | Save weights to Google Drive | ~30 s |
| 12 | — | Quick single-claim inference smoke test | ~20 s |

---

## Speed / memory tips

- **Smoke run first:** in Cell 4, use `--sample 3000` to build a tiny dataset,
  then run Cell 6 to confirm the pipeline works before the full ~90k-row run.
- **T4 out of memory in Phase 5?** Lower the batch size / sequence length in
  `phase5_deberta.py` (`BATCH_SIZE = 16 → 8`, `MAX_LEN = 128 → 96`).
- **Mistral (Cell 10) OOMs on T4** — it needs ~16 GB for the 4-bit model plus
  training state. Use an **A100** runtime, or skip it (the DeBERTa scorer works
  without the LLM justification layer).
- Colab disconnects idle runtimes — **always run Cell 11** to persist weights to
  Drive before you step away.

## After Colab: use the weights locally

Download `models/` (Cell 11 saves to Drive, or Cell 11-alt downloads a zip),
then on your Mac drop them into `FakeNews_ML/models/` and run the API/demo:

```bash
pip install -r requirements.api.txt
uvicorn api.main:app --port 8000     # hardened API
python gradio_app.py --standalone    # or the demo UI
```

## Outputs produced

- `models/deberta_best.pt`, `models/deberta_tokenizer/`, `models/deberta_results.json`
- `models/baseline_tfidf.pkl`, `models/baseline_results.json`
- `models/calibration.json`, `models/speaker_profiles.json`
- `eda_output/` — reliability diagram, SHAP HTML + charts, speaker plots
