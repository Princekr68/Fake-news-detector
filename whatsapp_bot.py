#!/usr/bin/env python3
"""
whatsapp_bot.py — Modular Automated WhatsApp Fact-Checking Bot Service.

Directly aligned with `live_check.py`, `run_cli.py`, and `credibility.py`:
- Universal Input Handling (after `@verify`, `/verify`, or direct DM):
    1. Plain text statements / claims
    2. Long viral forwarded messages (boilerplate stripping + central claim extraction)
    3. Short rumors / news headlines
    4. Web URLs / Social media links (natively routed via `live_check.extract_from_url()`)
- Includes automated Counter-Truth statement for FAKE verdicts, Domain Trust
  badges, and automatic `logs/audit.jsonl` logging (`channel="whatsapp"`).
- Dual Webhook Mode for Student Demos (Twilio Sandbox for WhatsApp + ngrok):
    * Async Callback Mode (recommended): Set TWILIO_ACCOUNT_SID & TWILIO_AUTH_TOKEN
      to avoid Twilio's 15s HTTP timeout during live web scraping + NLI.
    * Synchronous TwiML Fallback: Works out-of-the-box with zero Twilio credentials.
    * JSON Endpoint (`POST /api/verify`): Ready for local testing or Node/Baileys.
"""

from __future__ import annotations

import os
import warnings

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

import argparse
import base64
import re
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

from run_cli import VerificationEngineAdapter, VerificationResult

# ---------------------------------------------------------------------------
# 1. Universal Input Parser (Text, Viral Forwards, Headlines, URLs)
# ---------------------------------------------------------------------------
TRIGGER_PREFIX_RE = re.compile(r"^\s*(?:@verify|/verify|!verify)\b[:\-\s]*", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

FORWARD_NOISE_PATTERNS = [
    re.compile(r"\bforwarded\s+many\s+times\b[!.,:\s]*", re.IGNORECASE),
    re.compile(r"\bplease\s+(?:share|forward)\b[^.!?\n]*[.!?\n]?", re.IGNORECASE),
    re.compile(r"\bshare\s+(?:this\s+)?(?:with|to)\s+(?:all|everyone|\d+)[^.!?\n]*[.!?\n]?", re.IGNORECASE),
    re.compile(r"\bforward\s+(?:this\s+)?(?:as\s+much|to\s+\d+|immediately)[^.!?\n]*[.!?\n]?", re.IGNORECASE),
    re.compile(r"\b(?:breaking\s*news|urgent\s*alert|important\s*notice|good\s*news|alert)\s*[:!\-]*", re.IGNORECASE),
    re.compile(r"\b(?:good\s+morning|good\s+evening|good\s+afternoon|namaste|jai\s+hind)\b[^.!?\n]*[.!?\n]?", re.IGNORECASE),
    re.compile(r"[🚨🔥⚠️📢‼️❗✅❌👉👇👆💯🙏]+"),
]

FACTUAL_CUE_RE = re.compile(
    r"\b(?:announced|confirmed|approved|banned|slashed|increased|died|passed\s+away|"
    r"won|lost|launched|free|subsidy|scheme|government|pm|minister|rbi|court|"
    r"police|who|official|effective|from\s+tomorrow|percent|%|\d{4})\b",
    re.IGNORECASE,
)


@dataclass
class ParsedWhatsAppInput:
    raw_message: str
    input_kind: str  # "URL" | "FORWARDED_TEXT" | "CLAIM"
    url: Optional[str] = None
    claim_text: str = ""
    fallback_caption: Optional[str] = None


def strip_trigger(message: str) -> str:
    return TRIGGER_PREFIX_RE.sub("", message or "").strip()


def extract_central_claim(long_text: str, max_chars: int = 260) -> str:
    """Distill a long viral WhatsApp forward into its central verifiable claim."""
    cleaned = long_text
    for pattern in FORWARD_NOISE_PATTERNS:
        cleaned = pattern.sub("\n", cleaned)

    cleaned = re.sub(r"(?:^|\n)\s*[!?,.:;\-•*]+\s*", "\n", cleaned)
    cleaned = re.sub(r"\s*[!]{2,}\s*", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip(" !?.,:;\n\t")

    sentences = [
        s.strip(" -•*:\t!?")
        for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        if len(s.strip(" -•*:\t!?")) >= 20
    ]
    if not sentences:
        return re.sub(r"\s+", " ", cleaned)[:max_chars].strip()
    if len(sentences) == 1:
        return sentences[0][:max_chars].strip()

    def _score(sent: str, idx: int) -> float:
        score = 0.0
        score += 2.5 * len(FACTUAL_CUE_RE.findall(sent))
        score += 1.5 * len(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", sent))
        score += 1.0 * len(re.findall(r"(?<!^)\b[A-Z][a-z]{2,}\b", sent))
        if idx == 0:
            score += 2.0
        elif idx == 1:
            score += 1.0
        words = len(sent.split())
        if 8 <= words <= 32:
            score += 1.5
        return score

    ranked = sorted(enumerate(sentences), key=lambda p: _score(p[1], p[0]), reverse=True)
    return ranked[0][1][:max_chars].strip()


def parse_whatsapp_input(raw_message: str) -> ParsedWhatsAppInput:
    """Classify and normalize ANY WhatsApp message into a routing payload."""
    body = strip_trigger(raw_message)
    if not body:
        return ParsedWhatsAppInput(raw_message=raw_message, input_kind="CLAIM", claim_text="")

    url_match = URL_RE.search(body)
    if url_match:
        url = url_match.group(0).rstrip(".,);!?'\"")
        caption = URL_RE.sub("", body).strip(" -:\n\t")
        return ParsedWhatsAppInput(
            raw_message=raw_message,
            input_kind="URL",
            url=url,
            claim_text=url,
            fallback_caption=extract_central_claim(caption) if len(caption) >= 15 else None,
        )

    if len(body) > 240 or body.count("\n") >= 3:
        return ParsedWhatsAppInput(
            raw_message=raw_message,
            input_kind="FORWARDED_TEXT",
            claim_text=extract_central_claim(body),
        )

    return ParsedWhatsAppInput(
        raw_message=raw_message,
        input_kind="CLAIM",
        claim_text=re.sub(r"\s+", " ", body).strip(),
    )


# ---------------------------------------------------------------------------
# 2. WhatsApp Message Formatter (Mobile-Friendly & <= 1550 chars)
# ---------------------------------------------------------------------------
def _verdict_badge(verdict: str) -> Tuple[str, str]:
    v = verdict.upper()
    if v.startswith("REAL"):
        return "✅", "REAL / SUPPORTED"
    if v.startswith("FAKE"):
        return "🛑", "FAKE / CONTRADICTED"
    return "⚠️", "UNVERIFIED"


def format_whatsapp_reply(
    result: VerificationResult,
    parsed: ParsedWhatsAppInput,
    max_sources: int = 3,
) -> str:
    emoji, verdict_label = _verdict_badge(result.verdict)
    conf_str = f" ({result.confidence * 100:.0f}% confidence)" if result.confidence > 0 else ""

    lines: List[str] = [f"{emoji} *Verdict:* {verdict_label}{conf_str}"]

    if parsed.input_kind == "FORWARDED_TEXT" and parsed.claim_text:
        lines.append(f"🔎 *Checked Claim:* _{parsed.claim_text[:180]}_")
    elif parsed.input_kind == "URL" and result.extracted_claim:
        lines.append(f"🔗 *Extracted Claim:* _{result.extracted_claim[:180]}_")

    if result.counter_truth:
        lines.append(f"\n📌 *Verified Fact:*\n{result.counter_truth}")

    summary = (result.summary or "").strip()
    if len(summary) > 260:
        summary = summary[:257].rsplit(" ", 1)[0] + "..."
    lines.append(f"\n🧠 *Why / Summary:*\n{summary}")

    if result.sources:
        lines.append("\n📰 *Verified Sources:*")
        for idx, src in enumerate(result.sources[:max_sources], start=1):
            pub = src.publisher or "Source"
            trust_tag = f" (Trust {src.trust_weight:.2f})"
            if src.url:
                lines.append(f"{idx}. *{pub}*{trust_tag} — {src.url}")
            else:
                lines.append(f"{idx}. *{pub}*{trust_tag}")
    else:
        lines.append("\n📰 *Verified Sources:* No corroborating live articles matched.")

    reply = "\n".join(lines)
    if len(reply) > 1550:
        reply = reply[:1547] + "..."
    return reply


# ---------------------------------------------------------------------------
# 3. Core Bot Service (Thread-Safe with Cached Verifier)
# ---------------------------------------------------------------------------
class WhatsAppFactCheckService:
    def __init__(self, engine: Optional[VerificationEngineAdapter] = None) -> None:
        self.engine = engine or VerificationEngineAdapter()
        self._lock = threading.Lock()

    def handle_message(self, incoming_text: str, require_trigger: bool = False) -> Optional[str]:
        if require_trigger and not TRIGGER_PREFIX_RE.match(incoming_text or ""):
            return None

        parsed = parse_whatsapp_input(incoming_text)
        if not parsed.claim_text:
            return (
                "👋 Send any claim, forwarded rumor, news headline, or article URL "
                "(e.g., *@verify India won the Cricket World Cup 2023*) and I'll check "
                "it against live verified sources."
            )

        target = parsed.url if parsed.input_kind == "URL" and parsed.url else parsed.claim_text
        try:
            with self._lock:
                result = self.engine.verify(target, channel="whatsapp")
                if (
                    parsed.input_kind == "URL"
                    and result.verdict == "UNVERIFIED"
                    and parsed.fallback_caption
                ):
                    result = self.engine.verify(parsed.fallback_caption, channel="whatsapp")
        except Exception as exc:
            return (
                "⚠️ *Verification Error*\n"
                f"Could not complete live verification right now ({exc}). "
                "Ensure SearXNG (`http://127.0.0.1:8888`) is running and try again."
            )

        return format_whatsapp_reply(result, parsed)


# ---------------------------------------------------------------------------
# 4. Twilio REST Async Reply Helper (Zero Extra SDK Required)
# ---------------------------------------------------------------------------
def _twilio_credentials() -> Optional[Tuple[str, str, str]]:
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    sender = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886").strip()
    if sid and token:
        return sid, token, sender
    return None


def send_twilio_whatsapp_message(to_number: str, body: str) -> bool:
    creds = _twilio_credentials()
    if not creds:
        return False
    sid, token, sender = creds
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    payload = urllib.parse.urlencode(
        {"From": sender, "To": to_number, "Body": body}
    ).encode("utf-8")
    auth = base64.b64encode(f"{sid}:{token}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _twiml_response(message_text: str) -> str:
    safe_body = xml_escape(message_text)
    return f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{safe_body}</Message></Response>'


# ---------------------------------------------------------------------------
# 5. Lightweight Flask Webhook Server (Backend API Only — No UI)
# ---------------------------------------------------------------------------
def create_app(require_trigger_in_groups: bool = False):
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError as exc:
        raise RuntimeError(
            "Flask is required to run the webhook server. Install it with: pip install flask"
        ) from exc

    app = Flask(__name__)
    service = WhatsAppFactCheckService()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "whatsapp-fact-check-bot"})

    @app.post("/webhook/whatsapp")
    def twilio_whatsapp_webhook():
        incoming_body = request.form.get("Body", "").strip()
        from_number = request.form.get("From", "").strip()

        if require_trigger_in_groups and not TRIGGER_PREFIX_RE.match(incoming_body):
            return Response(
                '<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                mimetype="application/xml",
            )

        if _twilio_credentials() and from_number:
            def _worker() -> None:
                reply = service.handle_message(incoming_body, require_trigger=False)
                if reply:
                    send_twilio_whatsapp_message(from_number, reply)

            threading.Thread(target=_worker, daemon=True).start()
            return Response(
                _twiml_response("🔍 Checking live verified sources... your verdict will arrive shortly."),
                mimetype="application/xml",
            )

        reply = service.handle_message(incoming_body, require_trigger=False) or ""
        return Response(_twiml_response(reply), mimetype="application/xml")

    @app.post("/api/verify")
    def json_verify_endpoint():
        data: Dict[str, Any] = request.get_json(silent=True) or {}
        message = str(data.get("message") or data.get("text") or "").strip()
        parsed = parse_whatsapp_input(message)
        reply = service.handle_message(message, require_trigger=False)
        return jsonify(
            {
                "input_kind": parsed.input_kind,
                "extracted_claim": parsed.claim_text,
                "url": parsed.url,
                "reply": reply,
            }
        )

    return app


# ---------------------------------------------------------------------------
# 6. CLI Entry Point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular WhatsApp Fact-Checking Bot Service (Twilio / Baileys Backend)."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    parser.add_argument(
        "--require-trigger",
        action="store_true",
        help="Only respond to messages prefixed with @verify or /verify.",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Dry-run a single WhatsApp message locally in the terminal.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.test:
        parsed = parse_whatsapp_input(args.test)
        print(f"[Parsed Input Kind] : {parsed.input_kind}")
        print(f"[Routing Target]    : {parsed.url or parsed.claim_text}")
        print("-" * 60)
        service = WhatsAppFactCheckService()
        print(service.handle_message(args.test, require_trigger=args.require_trigger))
        return 0

    app = create_app(require_trigger_in_groups=args.require_trigger)
    mode = "ASYNC (Twilio REST)" if _twilio_credentials() else "SYNC (Direct TwiML)"
    print(f"WhatsApp Fact-Check Webhook listening on http://{args.host}:{args.port}")
    print(f"Twilio Mode : {mode}")
    print("Endpoints   : POST /webhook/whatsapp  |  POST /api/verify  |  GET /health")
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())