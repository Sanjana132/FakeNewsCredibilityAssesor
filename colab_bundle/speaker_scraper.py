"""
╔══════════════════════════════════════════════════════════════════════════╗
║  SPEAKER SCRAPER — PolitiFact & Snopes Metadata                         ║
║  Fake News & Source Credibility Detector                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Scrapes METADATA ONLY (claim text, label, speaker, date, URL).         ║
║  Does NOT download article bodies or images.                             ║
║                                                                          ║
║  Compliance:                                                             ║
║   • Reads and respects robots.txt before scraping.                       ║
║   • 2 s rate limit between requests.                                     ║
║   • Only follows /fact-checks/ and /factchecks/ paths.                   ║
║   • No login / paywalled content.                                        ║
║                                                                          ║
║  Output:                                                                 ║
║    data/scraped_metadata.jsonl   — one JSON record per line              ║
║    data/faiss_evidence.csv       — flat CSV ready for FAISS indexing     ║
╚══════════════════════════════════════════════════════════════════════════╝

Install:
    pip install requests beautifulsoup4 lxml

Run:
    python speaker_scraper.py --source politifact --max 200
    python speaker_scraper.py --source snopes --max 100
    python speaker_scraper.py --build-faiss   # build FAISS index from CSV
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.robotparser
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin, urlparse

_HERE    = Path(__file__).resolve().parent
DATA_DIR  = _HERE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT_S = 2.0
HEADERS = {
    "User-Agent": (
        "FakeNewsResearchBot/1.0 "
        "(academic research; contact sanj18reddy@gmail.com)"
    )
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. ROBOTS.TXT GUARD
# ─────────────────────────────────────────────────────────────────────────────

class RobotsGuard:
    def __init__(self, base_url: str):
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(urljoin(base_url, "/robots.txt"))
        try:
            rp.read()
        except Exception:
            rp = None
        self._rp = rp
        self._agent = HEADERS["User-Agent"].split("/")[0]

    def can_fetch(self, url: str) -> bool:
        if self._rp is None:
            return True
        return self._rp.can_fetch(self._agent, url)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCRAPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, session) -> "requests.Response | None":
    try:
        r = session.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return r
    except Exception as e:
        print(f"  WARN: {url} → {e}")
        return None


def scrape_politifact(max_items: int = 200) -> Iterator[dict]:
    """
    Scrape PolitiFact's /factchecks/ listing pages (metadata only).
    Returns dicts with keys: source, url, claim, speaker, verdict,
                              verdict_score, date.
    """
    import requests
    from bs4 import BeautifulSoup

    base = "https://www.politifact.com"
    guard = RobotsGuard(base)
    session = requests.Session()

    VERDICT_SCORE = {
        "true":          1.0,
        "mostly-true":   0.8,
        "half-true":     0.6,
        "barely-true":   0.4,
        "false":         0.2,
        "pants-fire":    0.0,
        "pants on fire": 0.0,
    }

    seen, count = set(), 0
    page = 1
    while count < max_items:
        url = f"{base}/factchecks/?page={page}"
        if not guard.can_fetch(url):
            print(f"  robots.txt disallows {url} — stopping")
            break
        resp = _get(url, session)
        if resp is None:
            break
        soup = BeautifulSoup(resp.text, "lxml")
        cards = soup.select("article.m-statement")
        if not cards:
            break

        for card in cards:
            if count >= max_items:
                break
            try:
                a_tag = card.select_one("div.m-statement__quote a")
                if not a_tag:
                    continue
                claim_url  = urljoin(base, a_tag["href"])
                if claim_url in seen:
                    continue
                seen.add(claim_url)
                claim_text = a_tag.get_text(strip=True)

                speaker_tag = card.select_one("a.m-statement__name")
                speaker = speaker_tag.get_text(strip=True) if speaker_tag else ""

                verdict_tag = card.select_one("img.c-image__original")
                verdict_raw = ""
                if verdict_tag and verdict_tag.get("alt"):
                    verdict_raw = verdict_tag["alt"].lower().strip()
                score = VERDICT_SCORE.get(verdict_raw, 0.5)

                date_tag = card.select_one("span.m-statement__date")
                date_str = date_tag.get_text(strip=True) if date_tag else ""

                if len(claim_text) < 20:
                    continue

                yield {
                    "source":        "politifact",
                    "url":           claim_url,
                    "claim":         claim_text,
                    "speaker":       speaker,
                    "verdict":       verdict_raw,
                    "verdict_score": score,
                    "date":          date_str,
                }
                count += 1
            except Exception:
                continue

        page += 1
        time.sleep(RATE_LIMIT_S)

    print(f"  PolitiFact: {count} records scraped")


def scrape_snopes(max_items: int = 200) -> Iterator[dict]:
    """
    Scrape Snopes's /fact-check/ category listing (metadata only).
    Returns dicts with same schema as scrape_politifact.
    """
    import requests
    from bs4 import BeautifulSoup

    base = "https://www.snopes.com"
    guard = RobotsGuard(base)
    session = requests.Session()

    VERDICT_SCORE = {
        "true":            1.0,
        "mostly true":     0.8,
        "mixture":         0.6,
        "mostly false":    0.4,
        "false":           0.2,
        "legend":          0.2,
        "outdated":        0.5,
        "unproven":        0.5,
        "miscaptioned":    0.35,
        "satire":          0.1,
    }

    seen, count = set(), 0
    page = 1
    while count < max_items:
        url = f"{base}/fact-check/page/{page}/"
        if not guard.can_fetch(url):
            print(f"  robots.txt disallows {url} — stopping")
            break
        resp = _get(url, session)
        if resp is None:
            break
        soup = BeautifulSoup(resp.text, "lxml")
        articles = soup.select("article.wp-block-snopes-article-card")
        if not articles:
            # Try alternative selector
            articles = soup.select("div.article-card")
        if not articles:
            break

        for article in articles:
            if count >= max_items:
                break
            try:
                a_tag = (article.select_one("a.article-link") or
                         article.select_one("h2 a") or
                         article.select_one("a"))
                if not a_tag:
                    continue
                claim_url  = urljoin(base, a_tag["href"])
                if claim_url in seen or "/fact-check/" not in claim_url:
                    continue
                seen.add(claim_url)
                claim_text = a_tag.get_text(strip=True)

                verdict_tag = article.select_one("span.rating-label")
                verdict_raw = verdict_tag.get_text(strip=True).lower() \
                              if verdict_tag else "unproven"
                score = VERDICT_SCORE.get(verdict_raw, 0.5)

                date_tag = article.select_one("time")
                date_str = date_tag.get("datetime", "")[:10] \
                           if date_tag else ""

                if len(claim_text) < 20:
                    continue

                yield {
                    "source":        "snopes",
                    "url":           claim_url,
                    "claim":         claim_text,
                    "speaker":       "unknown",
                    "verdict":       verdict_raw,
                    "verdict_score": score,
                    "date":          date_str,
                }
                count += 1
            except Exception:
                continue

        page += 1
        time.sleep(RATE_LIMIT_S)

    print(f"  Snopes: {count} records scraped")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FAISS INDEX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_faiss_index(csv_path: Path = DATA_DIR / "faiss_evidence.csv",
                      index_path: Path = DATA_DIR / "faiss.index",
                      meta_path:  Path = DATA_DIR / "faiss_meta.json") -> None:
    """
    Embed claim texts with all-MiniLM-L6-v2 and build a FAISS IndexFlatIP
    (inner product → cosine similarity on normalised vectors).
    """
    try:
        import faiss
        import numpy as np
        import pandas as pd
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ERROR: pip install faiss-cpu sentence-transformers")
        return

    if not csv_path.exists():
        print(f"  {csv_path} not found — run scraper first")
        return

    df = pd.read_csv(csv_path)
    texts = df["claim"].fillna("").tolist()
    if not texts:
        print("  No texts to embed")
        return

    print(f"  Embedding {len(texts):,} claims with all-MiniLM-L6-v2…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    print(f"  FAISS index: {index.ntotal:,} vectors, dim={dim}")

    meta = df[["claim", "speaker", "verdict", "verdict_score",
               "url", "source", "date"]].to_dict(orient="records")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Saved: {index_path.name}, {meta_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Speaker / Fact-Check Scraper")
    ap.add_argument("--source",  choices=["politifact","snopes","all"],
                    default="all")
    ap.add_argument("--max",     type=int, default=200,
                    help="Max records per source")
    ap.add_argument("--build-faiss", action="store_true",
                    help="Build FAISS index from data/faiss_evidence.csv")
    args = ap.parse_args()

    print("=" * 60)
    print("  SPEAKER SCRAPER — robots.txt compliant, 2s rate limit")
    print("=" * 60)

    if args.build_faiss:
        build_faiss_index()
        return

    import csv

    out_jsonl = DATA_DIR / "scraped_metadata.jsonl"
    out_csv   = DATA_DIR / "faiss_evidence.csv"

    scrapers = []
    if args.source in ("politifact", "all"):
        scrapers.append(("politifact", scrape_politifact(args.max)))
    if args.source in ("snopes", "all"):
        scrapers.append(("snopes", scrape_snopes(args.max)))

    all_records = []
    with open(out_jsonl, "w") as jf:
        for src, generator in scrapers:
            print(f"\nScraping {src}…")
            for rec in generator:
                jf.write(json.dumps(rec) + "\n")
                all_records.append(rec)

    if all_records:
        import pandas as pd
        df = pd.DataFrame(all_records)
        df.to_csv(out_csv, index=False)
        print(f"\n  Saved: {len(all_records):,} records → {out_jsonl.name}")
        print(f"  Saved: {out_csv.name}")
        print("\n  Next: python speaker_scraper.py --build-faiss")
    else:
        print("\n  No records scraped (check network / robots.txt)")


if __name__ == "__main__":
    main()
