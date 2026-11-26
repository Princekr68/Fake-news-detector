from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit


# 1. Domain Credibility Engine & Canonical Publisher Deduplication

DOMAIN_WEIGHTS: Dict[str, float] = {
    # Tier 1: Official government, public-record & science institutions (0.95)
    "pib.gov.in": 0.95,
    "gov.in": 0.95,
    "nic.in": 0.95,
    "rbi.org.in": 0.95,
    "eci.gov.in": 0.95,
    "who.int": 0.95,
    "un.org": 0.95,
    "nasa.gov": 0.95,
    "isro.gov.in": 0.95,
    "nature.com": 0.95,
    "icc-cricket.com": 0.95,
    "bcci.tv": 0.95,
    # Tier 1B: Certified fact-checkers & global/national wire services (0.90 - 0.95)
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "ptinews.com": 0.93,
    "bbc.com": 0.93,
    "bbc.co.uk": 0.93,
    "altnews.in": 0.92,
    "boomlive.in": 0.92,
    "factchecker.in": 0.92,
    "factly.in": 0.92,
    "factcrescendo.com": 0.92,
    "newschecker.in": 0.92,
    "vishvasnews.com": 0.90,
    "snopes.com": 0.92,
    "politifact.com": 0.92,
    "fullfact.org": 0.92,
    # Tier 2: Established national & international newsrooms (0.80 - 0.90)
    "thehindu.com": 0.90,
    "indianexpress.com": 0.90,
    "espncricinfo.com": 0.90,
    "nytimes.com": 0.90,
    "theguardian.com": 0.88,
    "washingtonpost.com": 0.88,
    "thequint.com": 0.88,
    "ndtv.com": 0.85,
    "timesofindia.indiatimes.com": 0.85,
    "hindustantimes.com": 0.85,
    "indiatoday.in": 0.85,
    "livemint.com": 0.85,
    "economictimes.indiatimes.com": 0.85,
    "business-standard.com": 0.85,
    "deccanherald.com": 0.85,
    "telegraphindia.com": 0.85,
    "sciencedaily.com": 0.85,
    "earthsky.org": 0.85,
    "espn.in": 0.85,
    "espn.com": 0.85,
    "freepressjournal.in": 0.80,
    "theprint.in": 0.82,
    "scroll.in": 0.82,
    "aljazeera.com": 0.82,
    "aninews.in": 0.82,
    "space.com": 0.82,
    "news18.com": 0.80,
    "firstpost.com": 0.80,
    "en.wikipedia.org": 0.78,
    # Tier 3: Social media / user-generated platforms (low standalone authority)
    "youtube.com": 0.40,
    "x.com": 0.35,
    "twitter.com": 0.35,
    "reddit.com": 0.35,
    "facebook.com": 0.30,
    "instagram.com": 0.30,
    "wordpress.com": 0.30,
    "t.me": 0.25,
    "blogspot.com": 0.25,
}

# Mirror / regional subdomains that belong to the exact same publisher
CANONICAL_PUBLISHER_ALIASES: Dict[str, str] = {
    "bbc.co.uk": "bbc.com",
    "m.bbc.com": "bbc.com",
    "m.bbc.co.uk": "bbc.com",
    "m.timesofindia.com": "timesofindia.indiatimes.com",
    "m.economictimes.com": "economictimes.indiatimes.com",
    "sports.ndtv.com": "ndtv.com",
    "hindi.ndtv.com": "ndtv.com",
    "english.factcrescendo.com": "factcrescendo.com",
    "hindi.factcrescendo.com": "factcrescendo.com",
    "marathi.factcrescendo.com": "factcrescendo.com",
    "twitter.com": "x.com",
}

DEFAULT_UNVERIFIED_WEIGHT = 0.30


def normalize_domain(domain_or_url: str) -> str:
    raw = (domain_or_url or "").strip().lower()
    if "://" in raw:
        raw = (urlsplit(raw).hostname or "").lower()
    return raw.removeprefix("www.")


def canonical_publisher(domain_or_url: str) -> str:
    """Collapse mirror domains (e.g. bbc.co.uk -> bbc.com) so one publisher votes once."""
    dom = normalize_domain(domain_or_url)
    if dom in CANONICAL_PUBLISHER_ALIASES:
        return CANONICAL_PUBLISHER_ALIASES[dom]
    for alias, canon in CANONICAL_PUBLISHER_ALIASES.items():
        if dom.endswith("." + alias):
            return canon
    # Collapse language/section subdomains of known publishers (e.g. sports.ndtv.com)
    for known in DOMAIN_WEIGHTS:
        if dom == known or dom.endswith("." + known):
            return CANONICAL_PUBLISHER_ALIASES.get(known, known)
    return dom


def get_domain_credibility(domain_or_url: str) -> Tuple[float, str]:
    """Return `(trust_weight, tier_label)` for a domain or URL."""
    dom = normalize_domain(domain_or_url)
    if not dom:
        return DEFAULT_UNVERIFIED_WEIGHT, "Unverified"

    canon = canonical_publisher(dom)
    if dom in DOMAIN_WEIGHTS:
        w = DOMAIN_WEIGHTS[dom]
    elif canon in DOMAIN_WEIGHTS:
        w = DOMAIN_WEIGHTS[canon]
    elif dom.endswith((".gov.in", ".nic.in", ".gov", ".edu", ".ac.in", ".mil")):
        w = 0.95
    else:
        w = DEFAULT_UNVERIFIED_WEIGHT

    if w >= 0.90:
        tier = "High Authority"
    elif w >= 0.75:
        tier = "Reputable News"
    elif w >= 0.50:
        tier = "Moderate"
    else:
        tier = "Unverified / Low"
    return w, tier


def score_weighted_consensus(
    rows: List[Dict[str, Any]],
    base_verdict: str,
    base_reason: str,
) -> Dict[str, Any]:
    """
    Apply domain credibility weights to `live_check.py`'s `rows` list.
    Deduplicates by canonical publisher (e.g. bbc.com + bbc.co.uk count once)
    and computes a credibility-weighted consensus & confidence score.
    """
    if not rows:
        return {
            "verdict": base_verdict,
            "reason": base_reason,
            "confidence": 0.0,
            "avg_credibility": 0.0,
            "enriched_rows": [],
        }

    by_publisher: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        item = r.get("item", {})
        raw_dom = normalize_domain(item.get("source") or item.get("link") or "")
        pub_key = canonical_publisher(raw_dom)
        trust_w, tier = get_domain_credibility(raw_dom)
        raw_score = 0.98 if r.get("kind") == "explicit" else float(r.get("score") or 0.70)
        weighted_power = round(trust_w * raw_score, 4)

        row_copy = dict(r)
        row_copy["canonical_publisher"] = pub_key
        row_copy["trust_weight"] = trust_w
        row_copy["trust_tier"] = tier
        row_copy["effective_score"] = raw_score
        row_copy["weighted_power"] = weighted_power

        # Keep only the strongest passage per canonical publisher
        if pub_key not in by_publisher or weighted_power > by_publisher[pub_key]["weighted_power"]:
            by_publisher[pub_key] = row_copy

    deduped = list(by_publisher.values())
    ent_weight = sum(r["weighted_power"] for r in deduped if r.get("label") == "entailment")
    con_weight = sum(r["weighted_power"] for r in deduped if r.get("label") == "contradiction")
    total_weight = ent_weight + con_weight
    avg_cred = sum(r["trust_weight"] for r in deduped) / max(1, len(deduped))

    verdict = base_verdict
    reason = base_reason

    if total_weight > 0:
        if con_weight > ent_weight and base_verdict != "UNVERIFIED":
            verdict = "FAKE / CONTRADICTED"
        elif ent_weight > con_weight and base_verdict != "UNVERIFIED":
            verdict = "REAL / SUPPORTED"
        elif base_verdict == "UNVERIFIED" and ent_weight > 0 and con_weight > 0:
            top_con = max((r["effective_score"] for r in deduped if r.get("label") == "contradiction"), default=0.0)
            top_ent = max((r["effective_score"] for r in deduped if r.get("label") == "entailment"), default=0.0)
            # Case A: One side has decisive NLI confidence (>= 0.85) while the other is weak (< 0.70)
            if top_ent >= 0.85 and top_con < 0.70:
                verdict = "REAL / SUPPORTED"
                reason = "High-confidence source directly supports this claim."
            elif top_con >= 0.85 and top_ent < 0.70:
                verdict = "FAKE / CONTRADICTED"
                reason = "High-confidence source directly contradicts this claim."
            # Case B: Both sides are confident; break tie by domain authority weight
            elif abs(con_weight - ent_weight) >= 0.05:
                if con_weight > ent_weight and top_con >= 0.70:
                    verdict = "FAKE / CONTRADICTED"
                    reason = "Higher-credibility sources contradict this claim."
                elif ent_weight > con_weight and top_ent >= 0.70:
                    verdict = "REAL / SUPPORTED"
                    reason = "Higher-credibility sources support this claim."

    if verdict.startswith("FAKE"):
        winning_rows = [r for r in deduped if r.get("label") == "contradiction"]
    elif verdict.startswith("REAL"):
        winning_rows = [r for r in deduped if r.get("label") == "entailment"]
    else:
        winning_rows = deduped

    if winning_rows:
        confidence = sum(r["effective_score"] for r in winning_rows) / len(winning_rows)
        if avg_cred < 0.45:
            confidence *= 0.80
        confidence = max(0.0, min(0.99, confidence))
    else:
        confidence = 0.0

    deduped.sort(key=lambda r: (r["weighted_power"], r.get("sim", 0.0)), reverse=True)
    return {
        "verdict": verdict,
        "reason": reason,
        "confidence": round(confidence, 4),
        "avg_credibility": round(avg_cred, 2),
        "enriched_rows": deduped,
    }


# 2. Automated Counter-Truth Explainer (1-sentence direct truth for FAKE)

_TRUTH_VERBS = re.compile(
    r"\b(?:defeated|beat|won|lost|announced|confirmed|clarified|denied|"
    r"attended|addressed|inaugurated|met|died|alive|passed\s+away|held|remains)\b",
    re.I,
)


def generate_counter_truth(
    claim: str,
    enriched_rows: List[Dict[str, Any]],
    verdict: str,
) -> Optional[str]:
    """
    When a claim is FAKE / CONTRADICTED, extract the cleanest, most direct
    1-sentence truth statement from the highest-credibility contradicting passage.
    """
    if not verdict.upper().startswith("FAKE"):
        return None

    contra_rows = [r for r in enriched_rows if r.get("label") == "contradiction"]
    if not contra_rows:
        return None

    claim_words = set(re.findall(r"[a-z0-9]{3,}", claim.lower()))
    best_sentence: Optional[str] = None
    best_score = -1.0

    for row in contra_rows[:3]:
        passage = str(row.get("passage") or "").strip()
        trust = float(row.get("trust_weight") or 0.5)
        for sent in re.split(r"(?<=[.!?।])\s+", passage):
            s = re.sub(r"\s+", " ", sent).strip(" \"'“”")
            if len(s) < 25 or len(s) > 260:
                continue
            words = set(re.findall(r"[a-z0-9]{3,}", s.lower()))
            overlap_ratio = len(claim_words & words) / max(1, len(claim_words))
            verb_bonus = 0.35 if _TRUTH_VERBS.search(s) else 0.0
            conciseness = 0.20 if 45 <= len(s) <= 170 else 0.0
            score = (trust * 1.2) + overlap_ratio + verb_bonus + conciseness
            if score > best_score:
                best_score = score
                best_sentence = s

    if not best_sentence:
        fallback = str(contra_rows[0].get("passage") or "").strip()
        best_sentence = fallback[:220].rsplit(" ", 1)[0] + ("..." if len(fallback) > 220 else "")

    if best_sentence and best_sentence[-1] not in ".!?।":
        best_sentence += "."
    return best_sentence



# 3. Structured Audit Logging 

DEFAULT_AUDIT_PATH = Path("logs") / "audit.jsonl"


def log_audit_entry(
    *,
    channel: str,
    query: str,
    input_type: str,
    verdict: str,
    confidence: float,
    avg_credibility: float,
    latency_s: float,
    articles_read: int,
    reason: str,
    counter_truth: Optional[str] = None,
    extracted_claim: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """Append a structured JSONL audit record to `logs/audit.jsonl`."""
    target = Path(log_path or os.getenv("AUDIT_LOG_PATH", str(DEFAULT_AUDIT_PATH)))
    target.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "channel": channel,
        "input_type": input_type,
        "query": query,
        "extracted_claim": extracted_claim,
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "avg_domain_credibility": round(avg_credibility, 2),
        "counter_truth": counter_truth,
        "reason": reason,
        "articles_read": articles_read,
        "latency_s": round(latency_s, 3),
        "sources": sources or [],
    }

    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target