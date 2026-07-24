---
title: Fake News & Source Credibility Detector
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# Fake News & Source Credibility Detector

Paste a news statement or article — the bot returns a **credibility score (0–1)**,
a verdict with a **90% confidence interval**, and **sources to check the claim
against** (disputing ones first for low-credibility claims).

Powered by a fine-tuned **DeBERTa-v3** regressor (test MAE 0.2512, ~13% better
than a TF-IDF baseline) with MC-Dropout uncertainty and Wikipedia / fact-check
source retrieval.

Code: https://github.com/Sanjana132/FakeNewsCredibilityAssesor
