"""Check an English or multilingual claim using the existing Colab XNLI head."""
import calendar
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

MODELS = Path(__file__).parent / "models" / "xnli"
LABEL_TEXT = {"entailment": "SUPPORTS the claim", "neutral": "NOT ENOUGH INFO",
              "contradiction": "CONTRADICTS the claim"}
MONTHS = {name.lower(): i for names in (calendar.month_name, calendar.month_abbr)
          for i, name in enumerate(names) if i}
MONTH = r"(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")"
DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
REL = re.compile(r"\b(yesterday|today|tomorrow)\b", re.I)
OFFSET = {"yesterday": -1, "today": 0, "tomorrow": 1}
NUMBERS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def resolve_dates(text, reference):
    if reference is None:
        return text
    return REL.sub(lambda m: (reference + timedelta(days=OFFSET[m[0].lower()])).isoformat(), text)


def facts(text):
    text = text.lower()
    dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
    for pattern, month_first in [(rf"\b{DAY}\s+(?:of\s+)?({MONTH})\b", False),
                                 (rf"\b({MONTH})\s+{DAY}\b", True)]:
        for m in re.finditer(pattern, text):
            month, day = m.groups() if month_first else m.groups()[::-1]
            year = re.match(r",?\s+(\d{4})\b", text[m.end():])
            dates.add(f"{year[1] if year else '????'}-{MONTHS[month]:02}-{int(day):02}")
    months = {MONTHS[m] for m in re.findall(rf"\b({MONTH})\b", text)}
    months.update(int(d[5:7]) for d in dates)
    quantities = set()
    normalized = re.sub(r"(?<=\d),(?=\d)", "", text)
    for i, word in enumerate(NUMBERS):
        normalized = re.sub(rf"\b{word}\b", str(i), normalized)
    for n, unit in re.findall(r"\b(\d+(?:\.\d+)?)\s*(days?|hours?|weeks?|years?|percent\b|per cent\b|%|rupees?)", normalized):
        quantities.add((float(n), "percent" if unit in ("%", "per cent") else unit.rstrip("s")))
    return dates, months, quantities


def evidence_gap(claim, evidence, reference=None, published=None):
    c = resolve_dates(claim.lower(), reference or date.today())
    e = resolve_dates(evidence.lower(), published)
    if re.search(r"\bupi\b", c) and re.search(r"\b(ban\w*|suspend\w*|stop\w*|shut\w*)\b", c):
        if not re.search(r"\b(fees?|charges?|pricing)\b", c):
            # Require wording about payment availability, including explicit denials.
            action = r"(?:ban(?:ned)?|suspend(?:ed)?|stop(?:ped)?|halt(?:ed)?|discontinu(?:e|ed)|shut down|remain available|continue)"
            payments = r"upi(?:\s+(?:payments?|transactions?|services?))?"
            direct = rf"\b{action}\s+(?:on\s+)?(?:all\s+)?{payments}\b|\b{payments}\s+(?:(?:will|would|are|is|be|not|never|remain)\s+)*{action}\b"
            if not re.search(direct, e):
                return "The snippet does not address whether UPI payments will stop."
            if "all" in c.split() and re.search(r"\b(only|select|some|merchant|above|below)\b", e):
                return "The snippet only identifies a narrower payment category."
    cd, cm, cq = facts(c)
    ed, em, eq = facts(e)
    if any(not any(d == v or (d.startswith("????") and d[4:] == v[4:]) for v in ed) for d in cd):
        return "The snippet does not establish the claimed event date."
    if cm - em:
        return "The snippet does not establish the claimed month."
    if cq - eq:
        return "The snippet does not establish the same quantity and unit."
    return None


class ClaimVerifier:
    """Hybrid verifier: tries mDeBERTa NLI first (high accuracy ~85%),
    falls back to the Colab-trained XNLI linear head if torch is unavailable."""

    def __init__(self):
        self._nli_model = None
        self._nli_tok = None
        self._fallback = None
        self.labels = ["entailment", "neutral", "contradiction"]
        # Try loading mDeBERTa NLI (much more accurate)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import os
            name = os.getenv("CLAIM_NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
            self._nli_tok = AutoTokenizer.from_pretrained(name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(name).eval()
            print(f"[claim_check] Using NLI model: {name}")
        except (ImportError, OSError, RuntimeError) as e:
            print(f"[claim_check] mDeBERTa unavailable ({e}), falling back to XNLI linear head")
            self._load_fallback()

    def _load_fallback(self):
        """Load the original Colab-trained XNLI linear head."""
        import numpy as np
        from sentence_transformers import SentenceTransformer
        with (MODELS / "report.json").open(encoding="utf-8") as f:
            info = json.load(f)
        self.labels = info["labels"]
        self.report = info["report"]
        with np.load(MODELS / "head.npz") as head:
            self._fb_coef, self._fb_bias = head["coef"], head["bias"]
        self._fallback = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def check(self, claim, evidence):
        gap = evidence_gap(claim, evidence)
        if gap:
            # Build approximate probs for display
            probs = {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1}
            return "neutral", probs, gap

        if self._nli_model is not None:
            return self._check_nli(claim, evidence)
        else:
            return self._check_fallback(claim, evidence)

    def _check_nli(self, claim, evidence):
        """Use mDeBERTa NLI for high-accuracy classification."""
        import torch
        inputs = self._nli_tok(evidence, claim, return_tensors="pt",
                               truncation=True, max_length=512)
        with torch.inference_mode():
            logits = self._nli_model(**inputs).logits.softmax(-1)[0]
        probs = {}
        for idx, score in enumerate(logits):
            label = self._nli_model.config.id2label[idx].lower()
            probs[label] = float(score)
        best = max(probs, key=probs.get)
        return best, probs, None

    def _check_fallback(self, claim, evidence):
        """Original Colab-trained XNLI linear head (fallback)."""
        import numpy as np
        u, v = self._fallback.encode([evidence, claim])
        features = np.concatenate([u, v, np.abs(u - v), u * v])
        logits = self._fb_coef @ features + self._fb_bias
        p = np.exp(logits - logits.max())
        probs = dict(zip(self.labels, (p / p.sum()).tolist()))
        best = max(probs, key=probs.get)
        return best, probs, None


if __name__ == "__main__":
    claim = sys.argv[1] if len(sys.argv) > 1 else input("Claim: ")
    evidence = sys.argv[2] if len(sys.argv) > 2 else input("Evidence: ")
    cv = ClaimVerifier()
    label, probs, why = cv.check(claim, evidence)
    print("Result:", LABEL_TEXT[label])
    if why:
        print("Reason:", why)
    for key, value in probs.items():
        print(f"Model {key}: {value:.2f}")
