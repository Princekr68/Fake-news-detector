from __future__ import annotations

import json
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

for _k, _v in {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "ABSL_MIN_LOG_LEVEL": "3",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTHONWARNINGS": "ignore",
}.items():
    os.environ.setdefault(_k, _v)
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request, send_file

from credibility import DEFAULT_AUDIT_PATH
from run_cli import VerificationEngineAdapter

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

# Single shared engine instance (low RAM: model loads once and stays warm)
engine = VerificationEngineAdapter(fast_mode=True)
_engine_lock = threading.Lock()


def _resolve_html_path() -> Path:
    candidates = [
        BASE_DIR / "templates" / "index.html",
        BASE_DIR / "index.html",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find `templates/index.html` or `index.html` in the project folder."
    )


def _read_recent_history(limit: int = 15) -> List[Dict[str, Any]]:
    audit_path = Path(os.getenv("AUDIT_LOG_PATH", str(DEFAULT_AUDIT_PATH)))
    if not audit_path.is_absolute():
        audit_path = BASE_DIR / audit_path
    if not audit_path.is_file():
        return []

    records: List[Dict[str, Any]] = []
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
                if len(records) >= limit:
                    break
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return records


@app.get("/")
def index():
    return send_file(_resolve_html_path())


@app.get("/api/history")
def api_history():
    return jsonify({"items": _read_recent_history(limit=15)})


@app.delete("/api/history")
def api_clear_history():
    audit_path = Path(os.getenv("AUDIT_LOG_PATH", str(DEFAULT_AUDIT_PATH)))
    if not audit_path.is_absolute():
        audit_path = BASE_DIR / audit_path
    try:
        if audit_path.is_file():
            audit_path.write_text("", encoding="utf-8")
    except OSError:
        pass
    engine._query_cache.clear()
    return jsonify({"status": "cleared"})


@app.post("/api/verify")
def api_verify():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("query") or "").strip()
    mode = str(payload.get("mode") or "live").strip().lower()  # "live" | "dataset"

    if not text:
        return jsonify({"error": "Please enter a claim, headline, or URL."}), 400

    if mode == "dataset":
        t0 = time.perf_counter()
        with _engine_lock:
            if "|" in text:
                c_part, e_part = [p.strip() for p in text.split("|", 1)]
                pretty, probs, why = engine.check_offline_evidence(c_part, e_part)
                verdict = (
                    "REAL"
                    if "SUPPORT" in pretty
                    else ("FAKE" if "CONTRADICT" in pretty else "UNVERIFIED")
                )
                return jsonify(
                    {
                        "mode": "dataset",
                        "submode": "offline_nli",
                        "query": text,
                        "verdict": verdict,
                        "headline": pretty,
                        "reason": why or "Evaluated offline using Claim vs. Evidence NLI.",
                        "confidence": round(max(probs.values()) if probs else 0.0, 4),
                        "probs": probs,
                        "sources": [],
                        "latency_s": round(time.perf_counter() - t0, 3),
                    }
                )

            style_scores = engine.analyze_style(text)
            if not style_scores:
                return jsonify(
                    {
                        "error": "No offline Colab models found in `models/isot/` or `models/liar/`."
                    }
                ), 404

            # Combine LIAR + ISOT SVM margins for top-level badge
            margins = [v["svm_margin"] for v in style_scores.values()]
            avg_margin = sum(margins) / len(margins)
            verdict = "FAKE" if avg_margin > 0 else "REAL"
            headline = (
                "Suspicious writing style detected by offline dataset models."
                if avg_margin > 0
                else "Normal journalistic writing style detected by offline dataset models."
            )
            return jsonify(
                {
                    "mode": "dataset",
                    "submode": "style_ml",
                    "query": text,
                    "verdict": verdict,
                    "headline": headline,
                    "reason": "Evaluates lexical style only (ISOT & LIAR TF-IDF models). Switch to Live Search for factual web verification.",
                    "style_models": style_scores,
                    "sources": [],
                    "latency_s": round(time.perf_counter() - t0, 3),
                }
            )

    # Default: Live Search (Mode B)
    try:
        with _engine_lock:
            res = engine.verify(text, channel="web_ui", audit=True)
    except Exception as exc:
        return jsonify({"error": f"Verification failed: {exc}"}), 500

    short_verdict = (
        "REAL"
        if res.verdict.startswith("REAL")
        else ("FAKE" if res.verdict.startswith("FAKE") else "UNVERIFIED")
    )

    # Choose crisp editorial headline: Counter-Truth for FAKE, or top snippet/reason
    if short_verdict == "FAKE" and res.counter_truth:
        headline = res.counter_truth
    elif short_verdict == "REAL" and res.sources and res.sources[0].snippet:
        headline = res.sources[0].snippet
    else:
        headline = res.summary

    sources_out = [
        {
            "publisher": s.publisher,
            "url": s.url,
            "title": s.title,
            "snippet": s.snippet,
            "stance": s.stance,
            "published": s.published,
            "trust_weight": s.trust_weight,
            "trust_tier": s.trust_tier,
        }
        for s in res.sources[:4]
    ]

    social_hosts = ("x.com", "twitter.com", "facebook.com", "fb.com", "instagram.com", "threads.net")
    if (
        short_verdict == "UNVERIFIED"
        and res.input_type == "URL"
        and any(h in res.query.lower() for h in social_hosts)
        and not res.extracted_claim
    ):
        headline = "This social media link is behind a login wall."
        res.summary = "Platforms like X, Facebook, and Instagram block automated scrapers on login-walled posts. Copy and paste the post text directly, or paste a public news article URL (BBC, The Hindu, NDTV, PIB, etc.)."

    return jsonify(
        {
            "mode": "live",
            "query": res.query,
            "input_type": res.input_type,
            "extracted_claim": res.extracted_claim,
            "verdict": short_verdict,
            "verdict_full": res.verdict,
            "headline": headline,
            "reason": res.summary,
            "confidence": round(res.confidence, 4),
            "avg_credibility": res.avg_credibility,
            "articles_read": res.articles_read,
            "latency_s": round(res.latency_s, 2),
            "cached": res.cached,
            "web_error": res.web_error,
            "sources": sources_out,
        }
    )


if __name__ == "__main__":
    print("Starting Fake News Detector Web UI on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)