"""
Chatbot Demo — thin UI over the agent pipeline.

This file is PRESENTATION ONLY: it calls agent.pipeline.assess(claim) → Verdict
and formats the result as chat markdown. All decision logic (gate, scoring,
retrieval, NLI stance, tier selection) lives in agent/pipeline.py so the API and
any other front-end share the exact same brain.

Run:
    MODEL_DEVICE=cpu python gradio_app.py
    python gradio_app.py --share
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import gradio as gr

from agent.pipeline import assess, is_factcheck


# ── verdict labels ──────────────────────────────────────────────────────────────
def _verdict(score: float) -> tuple[str, str]:
    if score < 0.35:
        return "🔴", "Likely False"
    if score < 0.65:
        return "🟡", "Uncertain / Mixed"
    return "🟢", "Likely Credible"


def _evidence_verdict(score: float) -> tuple[str, str]:
    """Label for a verdict CONFIRMED against fact-checks by the stance model."""
    if score < 0.35:
        return "🔴", "False — per fact-checkers"
    if score < 0.65:
        return "🟡", "Mixed / Partly true — per fact-checkers"
    return "🟢", "True — per fact-checkers"


def _domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url or "")
    return m.group(1) if m else ""


def _meter(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


# ── source / evidence formatters ────────────────────────────────────────────────
def _format_matches(matches: list[dict]) -> str:
    """Fact-checks the stance model confirmed address THIS claim."""
    lines = []
    for m in matches[:3]:
        who = _domain(m.get("url", "")) or m.get("source", "fact-check")
        sc = m.get("score")
        dot = " 🔴" if (sc is not None and sc < 0.4) else \
              " 🟢" if (sc is not None and sc > 0.6) else " 🟡"
        pct = int(round(m.get("nli_prob", 0.0) * 100))
        reviewed = (m.get("title") or "").strip()
        rating = (m.get("snippet") or "").strip()
        url = m.get("url") or ""
        link = f" — [link]({url})" if url else ""
        head = f"- **{who}**{dot} (match {pct}%) reviewed *“{reviewed[:80]}”*"
        if rating:
            head += f" → {rating[:150]}"
        lines.append(head + link)
    return "\n".join(lines)


def _format_factchecks(fc_srcs: list[dict]) -> str:
    """Retrieved fact-checks with their REAL rating + citation (unconfirmed match)."""
    lines = ["\n\n**⚖️ Published fact-checks on this topic** "
             "(read them — they may not all address your exact claim):"]
    for s in fc_srcs[:3]:
        who = _domain(s.get("url", "")) or s.get("source", "fact-check")
        sc = s.get("score")
        dot = " 🔴" if (sc is not None and sc < 0.4) else \
              " 🟢" if (sc is not None and sc > 0.6) else " 🟡"
        reviewed = (s.get("title") or "").strip()
        rating = (s.get("snippet") or "").strip()
        url = s.get("url") or ""
        link = f" — [link]({url})" if url else ""
        head = f"- **{who}**{dot}"
        if reviewed:
            head += f" reviewed *“{reviewed[:90]}”*"
        if rating:
            head += f" → {rating[:180]}"
        lines.append(head + link)
    return "\n".join(lines)


def _format_evidence(es, all_srcs: list[dict]) -> str:
    """Support/refute leaning from news/wiki snippets (soft signal)."""
    lean = {
        "support": "🟢 leans toward **supporting** this claim",
        "refute":  "🔴 leans toward **refuting** this claim",
        "mixed":   "🟡 is **mixed** on this claim",
    }[es.leaning]
    lines = [f"\n\n**🔎 Retrieved evidence {lean}** "
             f"({es.support} support / {es.refute} refute) — a soft signal from "
             "news/encyclopedia snippets, **not** a fact-check verdict:"]
    shown = set()
    for i in es.items[:4]:
        mark = "✅ supports" if i["stance"] == "support" else "❌ refutes"
        who = i.get("source", "web").replace("_", " ").upper()
        title = (i.get("title") or "").strip() or "(untitled)"
        url = i.get("url") or ""
        link = f" — [link]({url})" if url else ""
        pct = int(round(i.get("nli_prob", 0.0) * 100))
        lines.append(f"- {mark} ({pct}%) **[{who}]** {title}{link}")
        if url:
            shown.add(url)
    rest = [s for s in all_srcs if (s.get("url") or "") not in shown][:3]
    for s in rest:
        who = s.get("source", "web").replace("_", " ").upper()
        title = (s.get("title") or "").strip() or "(untitled)"
        url = s.get("url") or ""
        link = f" — [link]({url})" if url else ""
        lines.append(f"- **[{who}]** {title}{link}")
    return "\n".join(lines)


def _format_sources(bundle, low_credibility: bool, exclude_fc: bool = False) -> str:
    if bundle is None or not getattr(bundle, "sources", None):
        return ("\n\n_No external sources retrieved (offline, or rate-limited). "
                "Set `GOOGLE_FACTCHECK_API_KEY` / `NEWSAPI_KEY` or build the FAISS "
                "index for fact-check verdicts._")

    srcs = bundle.sources
    if exclude_fc:                       # fact-checks shown in their own panel
        srcs = [s for s in srcs if not is_factcheck(s)]
    if not srcs:
        return ""

    rated = [s for s in srcs if s.get("score") is not None]
    other = [s for s in srcs if s.get("score") is None]
    ordered = (rated + other)[:5]

    header = ("\n\n**🔎 Other sources to check this against:**"
              if low_credibility else "\n\n**🔎 Related sources:**")
    lines = [header]
    for s in ordered:
        tag = s.get("source", "web").replace("_", " ").upper()
        title = (s.get("title") or "").strip() or "(untitled)"
        url = s.get("url") or ""
        rating = ""
        if s.get("score") is not None:
            fc = s["score"]
            flag = "disputes" if fc < 0.4 else "supports" if fc > 0.6 else "mixed on"
            rating = f" · fact-check **{flag}** this ({fc:.2f})"
        link = f" — [link]({url})" if url else ""
        lines.append(f"- **[{tag}]** {title}{rating}{link}")
    return "\n".join(lines)


# ── respond: assess (the brain) → format (the UI) ──────────────────────────────
async def respond(message: str, history):
    v = await assess(message)

    if not v.ok:
        if v.reason == "__too_short__":
            return ("👋 Paste a **news statement or article** (a full sentence or "
                    "more) and I'll rate how credible it is and point you to sources "
                    "to check it.")
        return (f"🤔 I can only score **factual claims** — {v.reason}.\n\n"
                "Try a news-style statement, e.g. "
                "*“The unemployment rate fell to a record low last month.”*")

    if v.tier == "factcheck":
        emoji, label = _evidence_verdict(v.score)
        conf = int(round(v.confidence * 100))
        reply = [
            f"{emoji} **{label}** — verdict {v.score:.2f} / 1.00",
            f"`{_meter(v.score)}`",
            f"\n✅ **Confirmed against {v.n_matched} fact-check"
            f"{'s' if v.n_matched > 1 else ''}** a stance model verified address "
            f"*this* claim (entailment {conf}%):",
            _format_matches(v.matches),
        ]
        _, plabel = _verdict(v.prior)
        note = f"\n_Model's stylistic prior (secondary): {v.prior:.2f} — {plabel}._"
        if abs(v.prior - v.score) > 0.3:
            note += (" ⚠️ The style guess disagrees with the evidence — "
                     "**trust the fact-check.**")
        reply.append(note)
        reply.append(_format_sources(v.bundle, low_credibility=v.score < 0.5,
                                     exclude_fc=True))
    else:
        emoji, label = _verdict(v.prior)
        reply = [
            f"{emoji} **{label} — stylistic credibility {v.prior:.2f} / 1.00**",
            f"`{_meter(v.prior)}`  (90% CI {v.lo:.2f}–{v.hi:.2f})",
            "\n_This is the **model's stylistic estimate** — how the wording "
            "compares to historically true/false claims. It is **not** a fact-check, "
            "and the model has no knowledge of recent events._",
        ]
        if v.fc_srcs:
            reply.append(_format_factchecks(v.fc_srcs))
            if v.stance_available:
                reply.append("\n_The stance model couldn't confirm any of these "
                             "address your **exact** claim — read them yourself._")

        if v.tier == "evidence":
            reply.append(_format_evidence(v.evidence, v.other_srcs))
        else:
            if not v.fc_srcs:
                reply.append("\n\n_No published fact-check matched this claim — treat "
                             "it as **unverified** and weigh the sources below._")
            reply.append(_format_sources(v.bundle, low_credibility=v.prior < 0.5,
                                         exclude_fc=bool(v.fc_srcs)))

    reply.append(
        "\n\n_ℹ️ Confidence order: a **confirmed fact-check** verdict > a **support/"
        "refute leaning** from news/encyclopedia evidence > the model's **stylistic "
        "prior**. Always read the cited links._")
    return "\n".join(reply)


# ── UI ────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    "Vaccines contain microchips that let the government track people.",
    "The unemployment rate fell to a record low last month.",
    "Scientists have confirmed that drinking coffee cures cancer.",
    "The senate passed the infrastructure bill with bipartisan support.",
    "5G towers are spreading the virus across major cities.",
]


def build_ui() -> gr.ChatInterface:
    import inspect
    kwargs = dict(
        fn=respond,
        title="🔍 Fake News & Source Credibility Detector",
        description=(
            "Paste a **news statement or article**. The bot returns a credibility "
            "score (0 = false, 1 = credible), a verdict with a confidence interval, "
            "and sources to check the claim against. The score is a *stylistic* "
            "signal learned from historical fact-checks — **not a real-time "
            "fact-check** — so for current events, weigh the retrieved sources."
        ),
        examples=EXAMPLES,
    )
    if "type" in inspect.signature(gr.ChatInterface.__init__).parameters:
        kwargs["type"] = "messages"
    return gr.ChatInterface(**kwargs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Credibility chatbot demo")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Public Gradio link")
    ap.add_argument("--device", default=None, help="cpu / cuda / mps (overrides MODEL_DEVICE)")
    args = ap.parse_args()

    if args.device:
        os.environ["MODEL_DEVICE"] = args.device
    os.environ.setdefault("MODEL_DEVICE", "cpu")   # MPS NaNs for DeBERTa on Mac

    print(f"Starting credibility chatbot on port {args.port} "
          f"(device={os.environ['MODEL_DEVICE']})")
    build_ui().launch(server_port=args.port, share=args.share)
