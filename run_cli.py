
from __future__ import annotations

import os
import sys
import warnings


# 1. Pre-Import Environment Silencing 

_SILENT_ENV = {
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "ABSL_MIN_LOG_LEVEL": "3",
    "GLOG_minloglevel": "3",
    "GRPC_VERBOSITY": "ERROR",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_VERBOSITY": "error",
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "PYTHONWARNINGS": "ignore",
}
for _k, _v in _SILENT_ENV.items():
    os.environ.setdefault(_k, _v)

warnings.filterwarnings("ignore")

import argparse
import contextlib
import gc
import io
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from credibility import (
    generate_counter_truth,
    log_audit_entry,
    score_weighted_consensus,
)


def _silence_python_loggers() -> None:
    logging.disable(logging.CRITICAL)
    for name in (
        "transformers",
        "huggingface_hub",
        "sentence_transformers",
        "torch",
        "tensorflow",
        "absl",
        "urllib3",
        "requests",
        "charset_normalizer",
        "filelock",
    ):
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False


_silence_python_loggers()


@contextlib.contextmanager
def suppress_native_noise(suppress_stdout: bool = False):
    """Redirect OS-level stderr (and optionally stdout) to os.devnull."""
    saved_stderr_fd: Optional[int] = None
    saved_stdout_fd: Optional[int] = None
    devnull_fd: Optional[int] = None
    py_buf = io.StringIO()

    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            saved_stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
        except OSError:
            saved_stderr_fd = None

        if suppress_stdout:
            try:
                saved_stdout_fd = os.dup(1)
                os.dup2(devnull_fd, 1)
            except OSError:
                saved_stdout_fd = None

        with contextlib.redirect_stderr(py_buf):
            if suppress_stdout:
                with contextlib.redirect_stdout(py_buf):
                    yield
            else:
                yield
    finally:
        if saved_stderr_fd is not None:
            try:
                os.dup2(saved_stderr_fd, 2)
            finally:
                os.close(saved_stderr_fd)
        if saved_stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, 1)
            finally:
                os.close(saved_stdout_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)


# ---------------------------------------------------------------------------
# 2. Terminal Styling (ANSI with Windows VT100 support & plain-text fallback)
# ---------------------------------------------------------------------------
def _enable_windows_ansi() -> bool:
    if os.name != "nt":
        return sys.stdout.isatty()
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            return True
    except Exception:
        pass
    return False


USE_COLOR = _enable_windows_ansi() and not os.environ.get("NO_COLOR")


class C:
    RESET = "\033[0m" if USE_COLOR else ""
    BOLD = "\033[1m" if USE_COLOR else ""
    DIM = "\033[2m" if USE_COLOR else ""
    RED = "\033[91m" if USE_COLOR else ""
    GREEN = "\033[92m" if USE_COLOR else ""
    YELLOW = "\033[93m" if USE_COLOR else ""
    CYAN = "\033[96m" if USE_COLOR else ""


# 3. Result Schema & Unified Adapter (Mode A + Mode B)


_VIRAL_FORWARD_RE = re.compile(
    r"(?:sabko\s+forward|forward\s+karein|share\s+karein|sabhi\s+group|"
    r"forward\s+to\s+(?:all|everyone|\d+)|share\s+(?:this\s+)?with\s+(?:all|everyone)|"
    r"aadhaar\s*(?:number|no)?\s*(?:aur|and)?\s*otp|otp\s+daal|link\s+par\s+(?:apna|click|apply)|"
    r"turant\s+(?:apply|share|forward)|आधार\s*नंबर\s*डालें|तुरंत\s*.*शेयर\s*करें)",
    re.I,
)
_CITED_AUTHORITY_RE = re.compile(
    r"\b(?:rbi|nasa|aiims|who|isro|pradhan\s+mantri|pm\s+modi|government|govt|supreme\s+court)\b|"
    r"प्रधानमंत्री|आरबीआई",
    re.I,
)
_SENSATIONAL_HOAX_RE = re.compile(
    r"(?:band\s+ho\s+jayenge|zero\s+ho\s+jayegi|hawa\s+mein\s+udne|"
    r"7\s+din\s+mein|hamesha\s+ke\s+liye\s+(?:utar|chhod|sanyas)|"
    r"free\s+laptop|muft\s+laptop|मुफ्त\s+लैपटॉप|free\s+recharge|"
    r"raat\s+12\s+baje|gravity.*zero|chashma.*utar|sanyas\s+lene\s+ka\s+ailaan)",
    re.I,
)


def _detect_unverified_viral_hoax(text: str) -> Optional[Tuple[float, str]]:
    """
    Lightweight (0.1 ms, 0 MB RAM) conditional hoax guard that ONLY runs after
    live web search finds zero corroborating news (`UNVERIFIED`).
    """
    has_forward = bool(_VIRAL_FORWARD_RE.search(text))
    has_authority = bool(_CITED_AUTHORITY_RE.search(text))
    has_sensational = bool(_SENSATIONAL_HOAX_RE.search(text))

    if has_forward:
        return (
            0.86,
            "No official news corroborated this claim, and it contains classic WhatsApp viral-forward / phishing red flags.",
        )
    if has_authority and has_sensational:
        return (
            0.84,
            "No official announcement or reputable news supports this extraordinary claim attributed to a major authority.",
        )
    if has_sensational and len(text.split()) >= 10:
        return (
            0.78,
            "No reputable news corroborated this sensational viral rumor.",
        )
    return None


URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


@dataclass
class SourceCitation:
    publisher: str
    url: str
    title: str = ""
    snippet: str = ""
    stance: str = ""
    published: str = ""
    trust_weight: float = 0.30
    trust_tier: str = "Unverified / Low"


@dataclass
class VerificationResult:
    query: str
    input_type: str  # "URL" | "CLAIM"
    verdict: str  # "REAL / SUPPORTED" | "FAKE / CONTRADICTED" | "UNVERIFIED"
    confidence: float  # 0.0 - 1.0
    avg_credibility: float  # 0.0 - 1.0
    summary: str
    counter_truth: Optional[str] = None
    sources: List[SourceCitation] = field(default_factory=list)
    extracted_claim: Optional[str] = None
    style_signal: Optional[str] = None
    articles_read: int = 0
    web_error: Optional[str] = None
    model_warning: Optional[str] = None
    latency_s: float = 0.0
    cached: bool = False
    raw: Any = None


class VerificationEngineAdapter:
    """
    Silent, cached adapter connecting:
      - Mode B (ON):  `live_check.py` + `fast_mode.py` + `credibility.py`
      - Mode A (OFF): `style_check.py` (ISOT + LIAR) and `claim_check.py` (XNLI/NLI)
    """

    def __init__(
        self,
        fast_mode: bool = True,
        cache_size: int = 20,
        cache_ttl_s: float = 900.0,
    ) -> None:
        self.fast_mode = fast_mode
        self.cache_size = cache_size
        self.cache_ttl_s = cache_ttl_s
        self._live_mod: Any = None
        self._verifier: Any = None
        self._style_mod: Any = None
        self._style_models: Dict[str, Any] = {}
        self._claim_verifier: Any = None
        self._query_cache: "OrderedDict[str, Tuple[float, VerificationResult]]" = OrderedDict()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        with suppress_native_noise(suppress_stdout=True):
            try:
                import live_check  # type: ignore[import-not-found]

                self._live_mod = live_check
                if self.fast_mode:
                    try:
                        from fast_mode import apply_fast_optimizations

                        self._verifier = apply_fast_optimizations(live_check)
                    except ImportError:
                        self._verifier = live_check.Verifier()
                else:
                    self._verifier = live_check.Verifier()
            except ImportError as exc:
                raise RuntimeError(
                    "Could not import `live_check.py`. Run this script from your "
                    "project root (`C:\\my folder\\Fake news detector`)."
                ) from exc

        self._loaded = True

    def _ensure_style_models(self) -> Dict[str, Any]:
        """Lazy-load Colab-trained ISOT & LIAR NumPy models (~5 MB RAM)."""
        if self._style_models:
            return self._style_models
        try:
            import style_check  # type: ignore[import-not-found]

            self._style_mod = style_check
            for name in ("liar", "isot"):
                if (style_check.MODELS / name / "model.npz").exists():
                    self._style_models[name] = style_check.StyleModel(name)
        except Exception:
            self._style_models = {}
        return self._style_models

    def analyze_style(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Run Mode A (`style_check.py`) across available ISOT and LIAR models."""
        models = self._ensure_style_models()
        out: Dict[str, Dict[str, Any]] = {}
        for name, model in models.items():
            scores = model.score(text)
            scores["verdict"] = self._style_mod.verdict(scores)
            out[name] = scores
        return out

    def check_offline_evidence(self, claim: str, evidence: str) -> Tuple[str, Dict[str, float], Optional[str]]:
        """
        Run Mode A (`claim_check.py`) on a Claim + Evidence pair.
        Reuses the already-loaded mDeBERTa model when available to save RAM.
        """
        with suppress_native_noise(suppress_stdout=True):
            import claim_check  # type: ignore[import-not-found]

            if self._claim_verifier is None:
                if self._verifier is not None and getattr(self._verifier, "model", None) is not None:
                    cv = claim_check.ClaimVerifier.__new__(claim_check.ClaimVerifier)
                    cv._nli_model = self._verifier.model
                    cv._nli_tok = self._verifier.tok
                    cv._fallback = None
                    cv.labels = ["entailment", "neutral", "contradiction"]
                    self._claim_verifier = cv
                else:
                    self._claim_verifier = claim_check.ClaimVerifier()

            label, probs, why = self._claim_verifier.check(claim, evidence)
            pretty = claim_check.LABEL_TEXT.get(label, label.upper())
            return pretty, probs, why

    def _get_cached(self, key: str) -> Optional[VerificationResult]:
        entry = self._query_cache.get(key)
        if not entry:
            return None
        ts, res = entry
        if (time.monotonic() - ts) > self.cache_ttl_s:
            self._query_cache.pop(key, None)
            return None
        self._query_cache.move_to_end(key)
        return res

    def _put_cached(self, key: str, res: VerificationResult) -> None:
        self._query_cache[key] = (time.monotonic(), res)
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > self.cache_size:
            self._query_cache.popitem(last=False)

    def verify(
        self,
        user_input: str,
        include_style: bool = False,
        channel: str = "cli",
        audit: bool = True,
    ) -> VerificationResult:
        self.load()
        started = time.perf_counter()
        cleaned = user_input.strip().strip("\"'“”‘’").strip()
        cache_key = f"{cleaned.lower()}|style={include_style}"

        if self.fast_mode:
            cached_res = self._get_cached(cache_key)
            if cached_res is not None:
                cached_res.latency_s = time.perf_counter() - started
                cached_res.cached = True
                return cached_res

        is_link = bool(URL_PATTERN.fullmatch(cleaned))

        with suppress_native_noise(suppress_stdout=True):
            raw = self._live_mod.live_check(cleaned, cv=self._verifier)

        latency = time.perf_counter() - started
        extracted_claim = raw.get("extracted_claim")
        claim_for_analysis = extracted_claim or cleaned

        weighted = score_weighted_consensus(
            rows=raw.get("rows", []),
            base_verdict=raw.get("verdict", "UNVERIFIED"),
            base_reason=raw.get("reason", ""),
        )
        verdict = weighted["verdict"]
        confidence = weighted["confidence"]
        avg_credibility = weighted["avg_credibility"]
        reason = weighted["reason"]
        enriched_rows = weighted["enriched_rows"]

        # Conditional Viral-Hoax + Zero-Shot Fallback (ONLY when live search returned UNVERIFIED)
        if verdict == "UNVERIFIED" and not is_link and not raw.get("web_error"):
            hoax_hit = _detect_unverified_viral_hoax(claim_for_analysis)
            if hoax_hit is not None:
                verdict = "FAKE / CONTRADICTED"
                confidence, reason = hoax_hit
            elif hasattr(self._verifier, "classify_zero_shot_hoax"):
                try:
                    with suppress_native_noise(suppress_stdout=True):
                        is_hoax, hoax_score, _ = self._verifier.classify_zero_shot_hoax(claim_for_analysis)
                    if is_hoax:
                        verdict = "FAKE / CONTRADICTED"
                        confidence = round(hoax_score, 4)
                        reason = (
                            "No official news corroborated this claim, and zero-shot multilingual NLI "
                            "classified it as a likely viral hoax or scam."
                        )
                except Exception:
                    pass

        counter_truth = generate_counter_truth(
            claim=claim_for_analysis,
            enriched_rows=enriched_rows,
            verdict=verdict,
        )

        sources: List[SourceCitation] = []
        audit_sources = []
        for row in enriched_rows:
            item = row.get("item", {})
            citation = SourceCitation(
                publisher=str(item.get("source") or "Unknown Source"),
                url=str(item.get("link") or ""),
                title=str(item.get("title") or ""),
                snippet=str(row.get("passage") or ""),
                stance=str(row.get("label") or "").upper(),
                published=str(item.get("published") or ""),
                trust_weight=float(row.get("trust_weight") or 0.30),
                trust_tier=str(row.get("trust_tier") or "Unverified / Low"),
            )
            sources.append(citation)
            audit_sources.append(
                {
                    "publisher": citation.publisher,
                    "url": citation.url,
                    "stance": citation.stance,
                    "trust_weight": citation.trust_weight,
                    "passage": citation.snippet[:280],
                }
            )

        result = VerificationResult(
            query=cleaned,
            input_type="URL" if is_link else "CLAIM",
            verdict=verdict,
            confidence=confidence,
            avg_credibility=avg_credibility,
            summary=reason,
            counter_truth=counter_truth,
            sources=sources,
            extracted_claim=extracted_claim,
            articles_read=int(raw.get("articles_read") or 0),
            web_error=raw.get("web_error"),
            model_warning=raw.get("model_warning"),
            latency_s=latency,
            cached=False,
            raw=None,
        )

        if self.fast_mode and not result.web_error and result.verdict != "UNVERIFIED":
            self._put_cached(cache_key, result)

        if audit:
            try:
                log_audit_entry(
                    channel=channel,
                    query=result.query,
                    input_type=result.input_type,
                    verdict=result.verdict,
                    confidence=result.confidence,
                    avg_credibility=result.avg_credibility,
                    latency_s=result.latency_s,
                    articles_read=result.articles_read,
                    reason=result.summary,
                    counter_truth=result.counter_truth,
                    extracted_claim=result.extracted_claim,
                    sources=audit_sources,
                )
            except OSError:
                pass

        gc.collect()
        return result



# 4. Clean Terminal Rendering

def _confidence_bar(confidence: float, width: int = 20) -> str:
    filled = int(round(confidence * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {confidence * 100:5.1f}%"


def _verdict_style(verdict: str) -> tuple[str, str]:
    if verdict.startswith("REAL"):
        return C.GREEN, "REAL / SUPPORTED"
    if verdict.startswith("FAKE"):
        return C.RED, "FAKE / CONTRADICTED"
    return C.YELLOW, "UNVERIFIED"


def render_banner(live_mode: bool = True, fast_mode: bool = True) -> None:
    line = "=" * 72
    active = (
        f"{C.GREEN}LIVE ON (Mode B: SearXNG + RSS + NLI){C.RESET}"
        if live_mode
        else f"{C.YELLOW}OFFLINE (Mode A: Colab-Trained Models){C.RESET}"
    )
    print(f"{C.CYAN}{C.BOLD}{line}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  MULTI-MODE FAKE NEWS & REAL-TIME FACT VERIFICATION ENGINE (CLI){C.RESET}")
    print(f"  Active Mode: {active}")
    print(f"{C.CYAN}{C.BOLD}{line}{C.RESET}")
    print(
        f"{C.DIM}  Type 'on' = Live Web Mode  |  'off' = Offline Training Mode  |  "
        f":clear  |  exit{C.RESET}\n"
    )


def render_result(res: VerificationResult) -> None:
    color, label = _verdict_style(res.verdict)
    rule = "-" * 72
    print(f"\n{C.DIM}{rule}{C.RESET}")
    print(f"{C.BOLD}Mode         :{C.RESET} LIVE (Mode B)")
    print(f"{C.BOLD}Input Type   :{C.RESET} {res.input_type}")
    if res.extracted_claim and res.extracted_claim != res.query:
        print(f"{C.BOLD}Claim        :{C.RESET} {res.extracted_claim}")
    print(f"{C.BOLD}Verdict      :{C.RESET} {color}{C.BOLD}{label}{C.RESET}")
    print(f"{C.BOLD}Confidence   :{C.RESET} {color}{_confidence_bar(res.confidence)}{C.RESET}")
    if res.avg_credibility > 0:
        print(f"{C.BOLD}Source Trust :{C.RESET} {res.avg_credibility:.2f} / 1.00 (domain credibility average)")
    if res.counter_truth:
        print(f"{C.BOLD}Verified Fact:{C.RESET} {C.GREEN}{res.counter_truth}{C.RESET}")
    print(f"{C.BOLD}Reasoning    :{C.RESET} {res.summary}")

    if res.sources:
        print(f"\n{C.BOLD}Verified Sources ({len(res.sources)}):{C.RESET}")
        for idx, src in enumerate(res.sources[:5], start=1):
            pub_date = f" | {src.published}" if src.published else ""
            print(
                f"  {C.CYAN}{idx}. {src.publisher}{C.RESET} "
                f"{C.DIM}[{src.stance} | Trust {src.trust_weight:.2f} - {src.trust_tier}{pub_date}]{C.RESET}"
            )
            if src.snippet:
                preview = src.snippet if len(src.snippet) <= 180 else src.snippet[:177] + "..."
                print(f"     \"{preview}\"")
            if src.url:
                print(f"     {C.DIM}{src.url}{C.RESET}")
    else:
        print(f"\n{C.BOLD}Verified Sources:{C.RESET} {C.DIM}None matched for this query.{C.RESET}")

    if res.web_error:
        print(f"\n{C.YELLOW}Note:{C.RESET} {res.web_error}")
    if res.model_warning:
        print(f"{C.YELLOW}Note:{C.RESET} {res.model_warning}")

    cache_note = " (cached)" if res.cached else ""
    print(
        f"{C.DIM}Articles read: {res.articles_read}  |  "
        f"Completed in {res.latency_s:.2f}s{cache_note}{C.RESET}"
    )
    print(f"{C.DIM}{rule}{C.RESET}\n")


def render_offline_style(results: Dict[str, Dict[str, Any]], latency_s: float) -> None:
    rule = "-" * 72
    print(f"\n{C.DIM}{rule}{C.RESET}")
    print(f"{C.BOLD}Mode         :{C.RESET} OFFLINE TRAINING (Mode A — Colab TF-IDF Models)")
    if not results:
        print(f"{C.YELLOW}No trained models found in `models/isot/` or `models/liar/`.{C.RESET}")
    else:
        for name in ("liar", "isot"):
            if name not in results:
                continue
            s = results[name]
            is_sus = "SUSPICIOUS" in s["verdict"]
            col = C.RED if is_sus else C.GREEN
            print(f"\n  {C.CYAN}[{name.upper()} Model]{C.RESET}  {col}{C.BOLD}{s['verdict']}{C.RESET}")
            print(
                f"    Naive Bayes: {s['naive_bayes']:.2f}  |  "
                f"Logistic: {s['logistic']:.2f}  |  "
                f"SVM Margin: {s['svm_margin']:+.2f}  |  "
                f"Known Words: {s['known_words']}"
            )
        print(
            f"\n{C.DIM}Tip: For offline Claim vs. Evidence NLI, type: claim | evidence"
            f"  |  Completed in {latency_s * 1000:.1f}ms{C.RESET}"
        )
    print(f"{C.DIM}{rule}{C.RESET}\n")


def render_offline_evidence(pretty: str, probs: Dict[str, float], why: Optional[str], latency_s: float) -> None:
    rule = "-" * 72
    col = C.GREEN if "SUPPORT" in pretty else (C.RED if "CONTRADICT" in pretty else C.YELLOW)
    print(f"\n{C.DIM}{rule}{C.RESET}")
    print(f"{C.BOLD}Mode         :{C.RESET} OFFLINE TRAINING (Mode A — Claim vs. Evidence NLI)")
    print(f"{C.BOLD}Result       :{C.RESET} {col}{C.BOLD}{pretty}{C.RESET}")
    if why:
        print(f"{C.BOLD}Rule Check   :{C.RESET} {why}")
    score_str = "  |  ".join(f"{k}: {v:.2f}" for k, v in probs.items())
    print(f"{C.BOLD}Probabilities:{C.RESET} {score_str}")
    print(f"{C.DIM}Completed in {latency_s:.2f}s{C.RESET}")
    print(f"{C.DIM}{rule}{C.RESET}\n")



# 5. Entry Point 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Silent, low-RAM unified CLI with simple ON (Live) / OFF (Offline ML) switch."
    )
    parser.add_argument(
        "claim",
        nargs="?",
        help="Optional claim or URL to check in single-shot mode. Omit for interactive REPL.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Start in Offline Training Mode (Mode A: Colab-trained models).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Disable Low-RAM Fast Mode and use unpatched live_check.py defaults.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fast_enabled = not args.full
    live_mode = not args.offline
    engine = VerificationEngineAdapter(fast_mode=fast_enabled)

    # Single-shot CLI mode
    if args.claim:
        try:
            if not live_mode:
                t0 = time.perf_counter()
                if "|" in args.claim:
                    c_part, e_part = [p.strip() for p in args.claim.split("|", 1)]
                    pretty, probs, why = engine.check_offline_evidence(c_part, e_part)
                    render_offline_evidence(pretty, probs, why, time.perf_counter() - t0)
                else:
                    render_offline_style(engine.analyze_style(args.claim), time.perf_counter() - t0)
                return 0

            result = engine.verify(args.claim, channel="cli")
            render_result(result)
            return 0
        except Exception as exc:
            print(f"{C.RED}Error:{C.RESET} {exc}", file=sys.stderr)
            return 1

    # Interactive REPL mode
    render_banner(live_mode=live_mode, fast_mode=fast_enabled)

    while True:
        prompt = (
            f"{C.GREEN}{C.BOLD}[LIVE: ON]{C.RESET} {C.CYAN}Enter claim or URL > {C.RESET}"
            if live_mode
            else f"{C.YELLOW}{C.BOLD}[OFFLINE: ML]{C.RESET} {C.CYAN}Enter text (or claim | evidence) > {C.RESET}"
        )
        try:
            raw_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Exiting verification CLI.{C.RESET}")
            break

        if not raw_input:
            continue

        lower = raw_input.lower()
        if lower in {"exit", "quit", ":q", "q"}:
            print(f"{C.DIM}Exiting verification CLI.{C.RESET}")
            break
        if lower == ":clear":
            os.system("cls" if os.name == "nt" else "clear")
            render_banner(live_mode=live_mode, fast_mode=fast_enabled)
            continue
        if lower in {"on", ":on", "live"}:
            live_mode = True
            print(f"{C.GREEN}Switched to LIVE Mode (Mode B: Real-time web verification).{C.RESET}\n")
            continue
        if lower in {"off", ":off", "offline"}:
            live_mode = False
            print(
                f"{C.YELLOW}Switched to OFFLINE Mode (Mode A: Colab-trained ISOT/LIAR & XNLI models).{C.RESET}\n"
            )
            continue

        # Execute in active mode
        if not live_mode:
            t0 = time.perf_counter()
            try:
                if "|" in raw_input:
                    c_part, e_part = [p.strip() for p in raw_input.split("|", 1)]
                    print(f"{C.DIM}Running offline Claim vs. Evidence NLI...{C.RESET}", end="", flush=True)
                    pretty, probs, why = engine.check_offline_evidence(c_part, e_part)
                    print("\r" + " " * 48 + "\r", end="")
                    render_offline_evidence(pretty, probs, why, time.perf_counter() - t0)
                else:
                    results = engine.analyze_style(raw_input)
                    render_offline_style(results, time.perf_counter() - t0)
            except Exception as exc:
                print(f"\r{C.RED}Offline mode error:{C.RESET} {exc}\n")
        else:
            print(f"{C.DIM}Verifying against live sources...{C.RESET}", end="", flush=True)
            try:
                result = engine.verify(raw_input, channel="cli")
                print("\r" + " " * 40 + "\r", end="")
                render_result(result)
            except Exception as exc:
                print(f"\r{C.RED}Verification error:{C.RESET} {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())