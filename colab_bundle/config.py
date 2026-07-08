"""
Single source of truth for all project-wide constants.
Import this module wherever a constant is needed; never repeat magic numbers inline.
"""
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    root:          Path = _HERE
    data:          Path = _HERE / "data"
    models:        Path = _HERE / "models"
    eda_output:    Path = _HERE / "eda_output"
    archive:       Path = _HERE / "archive"
    utils:         Path = _HERE / "utils"
    tests:         Path = _HERE / "tests"


@dataclass(frozen=True)
class DataConfig:
    # Random seed — used in every shuffle/split/sample call
    seed:          int   = 42

    # Dataset identifiers (HuggingFace)
    liar2_id:      str   = "chengxuphd/liar2"
    multifc_id:    str   = "pszemraj/multi_fc"
    fever_id:      str   = "lucadiliello/fever"
    averitec_id:   str   = "pminervini/averitec"

    # FEVER row cap (prevents dataset dominance)
    fever_cap:     int   = 50_000

    # Train / val / test split proportions
    val_size:      float = 0.10
    test_size:     float = 0.10

    # Minimum text length after cleaning (chars)
    min_text_len:  int   = 20

    # Bayesian shrinkage prior sample size (context priors)
    prior_kappa:   float = 10.0


@dataclass(frozen=True)
class ModelConfig:
    # DeBERTa (Phase 5)
    deberta_name:    str   = "microsoft/deberta-v3-base"
    max_len:         int   = 128
    batch_size:      int   = 16
    epochs:          int   = 8
    lr:              float = 2e-5
    warmup_frac:     float = 0.15
    freeze_n_layers: int   = 4
    dropout:         float = 0.1
    mc_passes:       int   = 20
    patience:        int   = 3
    alpha_mse:       float = 0.7    # weight of MSE in combined loss
    alpha_mae:       float = 0.3    # weight of MAE in combined loss
    head_lr_mult:    float = 5.0    # head LR = lr * head_lr_mult

    # TF-IDF baseline (Phase 4)
    tfidf_max_features:  int = 50_000
    tfidf_ngram_range:   tuple = (1, 2)
    ridge_cv_alphas:     tuple = (0.01, 0.1, 1.0, 10.0, 100.0)
    max_shap_rows:       int  = 2_000

    # LLM fine-tune (Phase 7)
    llm_name:       str   = "mistralai/Mistral-7B-Instruct-v0.2"
    lora_r:         int   = 16
    lora_alpha:     int   = 32
    lora_dropout:   float = 0.05
    llm_epochs:     int   = 3
    llm_batch_size: int   = 4

    # Context encoder (Phase 6)
    context_embed_dim: int = 16


@dataclass(frozen=True)
class APIConfig:
    host:               str   = "0.0.0.0"
    port:               int   = 8000
    max_statement_len:  int   = 2_000
    rate_limit:         str   = "30/minute"
    request_timeout_s:  int   = 10
    redis_url:          str   = "redis://localhost:6379"
    ensemble_deberta_w: float = 0.7
    ensemble_tfidf_w:   float = 0.3


# Singleton instances — import these directly
PATHS  = Paths()
DATA   = DataConfig()
MODEL  = ModelConfig()
API    = APIConfig()

# Convenience re-exports so callers can write `from config import SEED`
SEED   = DATA.seed
