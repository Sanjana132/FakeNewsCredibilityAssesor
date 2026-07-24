# Deploying the chatbot to Hugging Face Spaces

The Gradio chatbot runs on a **free HF Space** (CPU). The 702 MB model is too
large for the Space git repo, so it lives in a separate **HF model repo** and the
app downloads it at startup.

Prerequisites: a free account at https://huggingface.co and the CLI:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login          # paste a WRITE token from hf.co/settings/tokens
```

Replace `Sanjana132` below with your HF username if different.

---

## 1 · Upload the trained model to the Hub

From the project root (where the `models/` folder lives):

```bash
huggingface-cli repo create credibility-detector-deberta --type model -y
huggingface-cli upload Sanjana132/credibility-detector-deberta ./models . --repo-type model
```

This pushes `deberta_best.pt`, `deberta_tokenizer/`, and `speaker_profiles.json`
(LFS is handled automatically). ~702 MB, one-time.

## 2 · Assemble the Space files

```bash
bash deploy/hf_space/build_space.sh      # → deploy/hf_space/build/
```

## 3 · Create the Space and push

```bash
huggingface-cli repo create credibility-detector --type space --space_sdk gradio -y

git clone https://huggingface.co/spaces/Sanjana132/credibility-detector space_repo
cp -r deploy/hf_space/build/* space_repo/
cd space_repo
git add .
git commit -m "Deploy credibility chatbot"
git push
```

The Space builds automatically (~5–10 min the first time) and goes live at
`https://huggingface.co/spaces/Sanjana132/credibility-detector`.

---

## Configuration (Space → Settings)

- **`MODEL_REPO`** (Variable) — only if your model repo name differs from the
  default `Sanjana132/credibility-detector-deberta`.
- **`HF_TOKEN`** (Secret) — only if you made the model repo **private**.
- Optional retrieval keys as Secrets for richer sources:
  `GOOGLE_FACTCHECK_API_KEY` (fact-check verdicts), `NEWSAPI_KEY`.

## Notes

- First request is slow (model loads into memory); subsequent ones are fast.
- On the free CPU tier each MC-Dropout prediction takes a few seconds — fine for
  a demo. Upgrade the Space hardware for lower latency.
- Alternative: skip the separate model repo and commit the weights into the Space
  itself via Git LFS. Simpler, but slower rebuilds and a heavier Space repo.
