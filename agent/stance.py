"""
Phase 10 — NLI Stance / Claim-Matching (Tier 2)

Tier-1 surfaced retrieved fact-checks but could NOT confirm a hit was about the
*exact* claim, so it refused to fabricate an aggregate verdict. Tier 2 adds that
confirmation with a cross-encoder NLI model, which lets a matched fact-check's
verdict transfer reliably.

Why a cross-encoder NLI model (measured, not assumed):
  claim = "5G towers spread COVID-19"
    entail=0.90  "coronavirus is the impact of 5G exposure"   (same claim)   ✅
    entail=1.00  "5G networks spread the coronavirus"         (same claim)   ✅
    neutral=1.0  "China was first to have 100,000 5G towers"  (same TOPIC)   ✅ rejected
    contra=1.0   "Bill Gates created the new coronavirus"     (different)    ✅ rejected
  A bi-encoder rated the WRONG claim (0.595) above the right one (0.588) — unusable.

Verdict transfer (ENTAILMENT-ONLY, by measurement):
  • entails → user asserts the same thing the fact-checker reviewed → take its verdict V
  • neutral / contradiction → NOT used for the verdict (still shown as related)

Contradiction-inversion (invert V when the user claim contradicts a reviewed
claim) was tested and REJECTED: NLI marks two *different* false conspiracies as
"contradiction" (e.g. "5G spreads COVID" vs "Bill Gates created coronavirus",
contra=1.00), and inverting a False verdict there wrongly boosts credibility.
Contradiction between arbitrary claims ≠ logical negation, so it's unsafe.
Entailment is high-precision and matches how fact-checkers store claims (the
misinformation as-stated, rated False), so entailment-only is both safe and
sufficient.

Model: cross-encoder/nli-deberta-v3-base
  output label order: [contradiction, entailment, neutral]
Lazy-loaded + cached; degrades gracefully (available=False) if the model or the
sentence-transformers dependency is missing.

Install:
    pip install sentence-transformers
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_CONTRA, _ENTAIL, _NEUTRAL = 0, 1, 2   # label order for this model
_MATCH_THRESHOLD = 0.5                  # min entail prob to accept a fact-check MATCH
_EVIDENCE_THRESHOLD = 0.6               # higher bar for noisy news/wiki snippets

_MODEL = None
_LOAD_TRIED = False


def _get_model():
    """Lazily load and cache the cross-encoder. Returns None if unavailable."""
    global _MODEL, _LOAD_TRIED
    if _LOAD_TRIED:
        return _MODEL
    _LOAD_TRIED = True
    try:
        from sentence_transformers import CrossEncoder
        _MODEL = CrossEncoder(_MODEL_NAME)
    except Exception as e:  # missing dep, no network, etc.
        print(f"[stance] NLI model unavailable ({e}); Tier-2 disabled.")
        _MODEL = None
    return _MODEL


@dataclass
class StanceResult:
    verdict:    Optional[float] = None   # aggregated 0-1 credibility, None if unmatched
    confidence: float           = 0.0    # mean NLI prob over matched fact-checks
    n_matched:  int             = 0
    matches:    list            = field(default_factory=list)  # matched src dicts + relation
    available:  bool            = True   # False → model couldn't load; caller falls back


def assess(claim: str, fc_srcs: list[dict]) -> StanceResult:
    """Confirm which retrieved fact-checks address `claim`, and transfer verdicts.

    Each element of `fc_srcs` is a SourceResult dict with at least 'title' (the
    reviewed claim) and 'score' (the fact-checker's verdict, 0-1).
    """
    model = _get_model()
    if model is None:
        return StanceResult(available=False)

    scored = [s for s in fc_srcs if s.get("score") is not None and (s.get("title") or "").strip()]
    if not scored:
        return StanceResult(available=True)

    # NLI is DIRECTIONAL: a specific reviewed claim ("tracks the patient's
    # location") isn't entailed by a general user claim ("tracks people"), but it
    # entails the user claim. Claim-equivalence is bidirectional, so we score both
    # directions and take the max entailment / max contradiction.
    p_fwd = model.predict([(claim, s["title"]) for s in scored], apply_softmax=True)
    p_rev = model.predict([(s["title"], claim) for s in scored], apply_softmax=True)

    matches: list[dict] = []
    weighted_sum = 0.0
    weight_total = 0.0
    for s, pf, pr in zip(scored, p_fwd, p_rev):
        entail = max(float(pf[_ENTAIL]), float(pr[_ENTAIL]))
        contra = max(float(pf[_CONTRA]), float(pr[_CONTRA]))
        V = float(s["score"])
        # Match only if the two claims mutually assert the same thing: strong
        # entailment in at least one direction, clearly above contradiction.
        if not (entail >= _MATCH_THRESHOLD and entail > contra):
            continue
        matches.append({**s, "relation": "entails", "nli_prob": entail,
                        "transferred": V})
        weighted_sum += V * entail
        weight_total += entail

    if not matches:
        return StanceResult(available=True)

    return StanceResult(
        verdict=weighted_sum / weight_total,
        confidence=weight_total / len(matches),
        n_matched=len(matches),
        matches=matches,
        available=True,
    )


@dataclass
class EvidenceStance:
    """Support/refute tally over evidence snippets that carry NO verdict of their
    own (news, Wikipedia) — a soft leaning, deliberately weaker than a matched
    fact-check verdict because short snippets make NLI noisy."""
    support:   int  = 0
    refute:    int  = 0
    items:     list = field(default_factory=list)   # src dicts + stance + nli_prob
    available: bool = True

    @property
    def leaning(self) -> Optional[str]:
        if self.support == 0 and self.refute == 0:
            return None
        if self.support > self.refute:
            return "support"
        if self.refute > self.support:
            return "refute"
        return "mixed"


def assess_evidence(claim: str, evidence_srcs: list[dict]) -> EvidenceStance:
    """NLI over non-fact-check snippets, FEVER direction (premise=evidence,
    hypothesis=claim): entailment ⇒ the evidence SUPPORTS the claim, contradiction
    ⇒ it REFUTES it. Used only when no fact-check was confirmed, and surfaced as a
    'leaning', never a hard verdict.
    """
    model = _get_model()
    if model is None:
        return EvidenceStance(available=False)

    def _text(s: dict) -> str:
        return f"{s.get('title', '')}. {s.get('snippet', '')}".strip(" .")

    srcs = [s for s in evidence_srcs if _text(s)]
    if not srcs:
        return EvidenceStance(available=True)

    probs = model.predict([(_text(s), claim) for s in srcs], apply_softmax=True)

    items: list[dict] = []
    support = refute = 0
    for s, p in zip(srcs, probs):
        contra, entail = float(p[_CONTRA]), float(p[_ENTAIL])
        if entail >= _EVIDENCE_THRESHOLD and entail > contra:
            stance, prob = "support", entail
            support += 1
        elif contra >= _EVIDENCE_THRESHOLD and contra > entail:
            stance, prob = "refute", contra
            refute += 1
        else:
            continue  # neutral snippet → no usable signal
        items.append({**s, "stance": stance, "nli_prob": prob})

    return EvidenceStance(support=support, refute=refute, items=items, available=True)
