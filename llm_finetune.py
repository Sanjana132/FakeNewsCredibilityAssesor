"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 7 — Mistral-7B-Instruct QLoRA Fine-Tuning                       ║
║  Fake News & Source Credibility Detector                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Trains Mistral-7B to generate evidence-grounded justification           ║
║  paragraphs for credibility verdicts.                                    ║
║                                                                          ║
║  Why Mistral-7B (not GPT-4o / Claude):                                  ║
║   • Runs locally — zero per-query cost after fine-tuning                 ║
║   • Adapter only ~80 MB (vs 14 GB full weights)                          ║
║   • Fine-tuned on DOMAIN DATA (fact-checking justifications), not        ║
║     generic RLHF — better calibration for this specific task             ║
║   • CUDA ONLY: BitsAndBytes NF4 has no MPS/CPU kernel                   ║
║                                                                          ║
║  Training data:                                                          ║
║   • Primary:   LIAR-2 justifications (~23k pairs)                        ║
║   • Secondary: AVeriTeC Q&A evidence chains (~4.5k pairs)                ║
║                                                                          ║
║  Prompt template (Mistral-Instruct):                                     ║
║    [INST] Claim: {text}                                                  ║
║    Speaker: {speaker} | Context: {context}                               ║
║    Verdict: {label}                                                      ║
║    Explain why this claim is {verdict}. [/INST]                          ║
║    {justification}                                                       ║
╚══════════════════════════════════════════════════════════════════════════╝

CUDA ONLY — run on Google Colab T4/A100 or local GPU.

Install:
    pip install transformers peft trl bitsandbytes accelerate sentencepiece
    pip install evaluate rouge-score bert-score

Run:
    python llm_finetune.py --train
    python llm_finetune.py --generate --claim "Obama doubled the deficit"
    python llm_finetune.py --eval-rouge   # ROUGE-L on val set
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_HERE     = Path(__file__).resolve().parent
DATA_DIR   = _HERE / "data"
MODEL_DIR  = _HERE / "models"

sys.path.insert(0, str(_HERE))
from utils.seed import set_seed
set_seed(42)

# ── constants ────────────────────────────────────────────────────────────────
BASE_MODEL     = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_DIR    = MODEL_DIR / "mistral_adapter"
MAX_SEQ_LEN    = 1024
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj"]
EPOCHS         = 3
BATCH_SIZE     = 4
GRAD_ACCUM     = 4
LR             = 2e-4
WARMUP_RATIO   = 0.05
RESPONSE_TMPL  = " [/INST]"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA PREP
# ─────────────────────────────────────────────────────────────────────────────

def _score_to_verdict(score: float) -> str:
    if score < 0.35:  return "false"
    if score < 0.65:  return "mixed/partially true"
    return "true"


def build_prompt(row: dict) -> str:
    """Build Mistral-Instruct prompt from a data row."""
    verdict = _score_to_verdict(float(row.get("credibility_score", 0.5)))
    speaker = str(row.get("speaker", "")).strip() or "an unnamed speaker"
    context = str(row.get("context", "unknown")).strip()
    label   = str(row.get("label_original", "")).strip()
    text    = str(row.get("text", "")).strip()
    just    = str(row.get("justification", "")).strip()

    instruction = (
        f"[INST] Claim: {text}\n"
        f"Speaker: {speaker} | Context: {context}\n"
        f"Verdict: {label}\n"
        f"Explain why this claim is {verdict}. [/INST]"
    )
    return f"{instruction} {just}"


def load_training_data():
    """Return HuggingFace Dataset of formatted prompts from LIAR-2 + AVeriTeC."""
    import pandas as pd
    from datasets import Dataset as HFDataset

    records = []
    for split in ("train", "val"):
        path = DATA_DIR / f"{split}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # Keep only rows with a usable justification (≥30 chars)
        has_just = (
            df["justification"].notna() &
            df["justification"].astype(str).str.len().ge(30) &
            df["dataset"].isin(["liar2", "averitec"])
        )
        df = df[has_just].copy()
        for _, row in df.iterrows():
            prompt = build_prompt(row.to_dict())
            if len(prompt) < 80:
                continue
            records.append({"text": prompt, "split": split})

    print(f"  LLM training records: {len(records):,}")
    return HFDataset.from_list(records)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _check_cuda():
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Mistral-7B QLoRA requires a GPU.")
        print("       Use Google Colab T4 (free) or any CUDA-capable machine.")
        sys.exit(1)
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu}  ({vram:.1f} GB VRAM)")


def load_base_model_and_tokenizer():
    """Load Mistral-7B in 4-bit NF4 with BitsAndBytes."""
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               BitsAndBytesConfig)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"  Loading {BASE_MODEL} in 4-bit NF4…")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def add_lora(model):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train():
    _check_cuda()

    from transformers import TrainingArguments
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    print("\nLoading data…")
    dataset = load_training_data()
    train_ds = dataset.filter(lambda x: x["split"] == "train")
    val_ds   = dataset.filter(lambda x: x["split"] == "val")
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    print("\nLoading base model…")
    model, tokenizer = load_base_model_and_tokenizer()
    model = add_lora(model)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TMPL,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(ADAPTER_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        seed=42,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        data_collator=collator,
        args=training_args,
    )

    print("\nTraining…")
    trainer.train()

    print("\nSaving adapter…")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"  Saved: models/mistral_adapter/  (~80 MB adapter weights only)")
    print("✓ LLM fine-tuning complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_adapter_model():
    """Load the fine-tuned adapter on top of 4-bit base for inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=False)
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_explanation(claim: str, speaker: str = "",
                          context: str = "unknown",
                          score: float = 0.5,
                          label: str = "",
                          max_new_tokens: int = 256) -> str:
    """
    Generate a fact-checking justification paragraph.

    This is called by the LangGraph REASON node (agent/graph.py) when
    score < 0.5 (i.e., the claim is flagged as likely false/misleading).
    """
    if not ADAPTER_DIR.exists():
        return (f"[LLM adapter not available — score {score:.2f}. "
                "Run python llm_finetune.py --train to fine-tune.]")

    import torch
    model, tokenizer = load_adapter_model()

    verdict = _score_to_verdict(score)
    row = {"text": claim, "speaker": speaker, "context": context,
           "credibility_score": score, "label_original": label,
           "justification": ""}
    prompt = build_prompt(row)

    # Trim to just the instruction part (before [/INST])
    instruction = prompt.split(RESPONSE_TMPL)[0] + RESPONSE_TMPL

    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def eval_rouge(n_samples: int = 200):
    """ROUGE-L evaluation on val-set justifications."""
    import pandas as pd
    from evaluate import load as eval_load

    rouge = eval_load("rouge")

    df = pd.read_csv(DATA_DIR / "val.csv")
    df = df[
        df["justification"].notna() &
        df["justification"].astype(str).str.len().ge(30) &
        df["dataset"].isin(["liar2", "averitec"])
    ].sample(min(n_samples, len(df)), random_state=42)

    predictions, references = [], []
    for _, row in df.iterrows():
        pred = generate_explanation(
            row["text"], row.get("speaker", ""),
            row.get("context", "unknown"),
            row.get("credibility_score", 0.5),
            row.get("label_original", ""),
        )
        predictions.append(pred)
        references.append(str(row["justification"]))

    scores = rouge.compute(predictions=predictions, references=references)
    print(f"\n  ROUGE-L : {scores['rougeL']:.4f}")
    print(f"  ROUGE-1 : {scores['rouge1']:.4f}")
    print(f"  ROUGE-2 : {scores['rouge2']:.4f}")

    results = {k: round(v, 4) for k, v in scores.items()}
    results["n_samples"] = n_samples
    (MODEL_DIR / "llm_eval.json").write_text(json.dumps(results, indent=2))
    print("  Saved: models/llm_eval.json")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Phase 7 — Mistral-7B QLoRA")
    ap.add_argument("--train",         action="store_true",
                    help="Fine-tune adapter (CUDA required)")
    ap.add_argument("--generate",      action="store_true",
                    help="Generate explanation for --claim")
    ap.add_argument("--claim",         default="",
                    help="Claim text for --generate")
    ap.add_argument("--speaker",       default="")
    ap.add_argument("--context",       default="unknown")
    ap.add_argument("--score",   type=float, default=0.25,
                    help="Credibility score for --generate")
    ap.add_argument("--eval-rouge",    action="store_true",
                    help="ROUGE-L on val set")
    ap.add_argument("--n-samples", type=int, default=200,
                    help="Samples for ROUGE eval")
    args = ap.parse_args()

    print("=" * 60)
    print("  PHASE 7 — Mistral-7B QLoRA Fine-Tuning")
    print("=" * 60)

    if args.train:
        train()
    elif args.generate:
        claim = args.claim or input("Claim: ")
        print("\nGenerating explanation…")
        explanation = generate_explanation(
            claim, args.speaker, args.context, args.score)
        print(f"\nExplanation:\n{explanation}")
    elif args.eval_rouge:
        eval_rouge(n_samples=args.n_samples)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
