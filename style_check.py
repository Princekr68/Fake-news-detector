import sys, re, json
from pathlib import Path
import numpy as np

MODELS = Path(__file__).parent / "models"      # folder with isot/ liar/ xnli/


def tokens(text):                               # same tokenizer as in training
    return re.findall(r"[a-z']+", text.lower())


def clean_isot(t):                              # same cleaning as ISOT training
    t = re.sub(r"^.{0,120}?\(Reuters\)\s*-?\s*", "", str(t))
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\breuters\b", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


class StyleModel:
    def __init__(self, name):
        folder = MODELS / name
        self.name = name
        self.vocab = json.load(open(folder / "vocab.json"))      # word -> column
        self.p = np.load(folder / "model.npz")                   # idf + NB / LR / SVM weights
        self.idf = self.p["idf"]

    def vectorize(self, text):                  # own TF-IDF, identical to training
        text = clean_isot(text) if self.name == "isot" else text
        x = np.zeros(len(self.vocab))
        for w in tokens(text):
            j = self.vocab.get(w)
            if j is not None:
                x[j] += 1                       # term frequency
        x = x * self.idf                        # tf * idf
        n = np.linalg.norm(x)
        return x / n if n else x                # L2 normalise

    def score(self, text):
        """Fake-style scores from the three models (higher = more suspicious)."""
        x = self.vectorize(text)
        nb = x @ self.p["nb_log_prob"].T + self.p["nb_log_prior"]          # log-prob of [real, fake]
        nb_fake = 1 / (1 + np.exp(np.clip(nb[0] - nb[1], -50, 50)))
        lr_fake = 1 / (1 + np.exp(-np.clip(x @ self.p["lr_coef"][0] + self.p["lr_bias"][0], -50, 50)))
        svm_margin = float(x @ self.p["svm_coef"][0] + self.p["svm_bias"][0])   # >0 means fake
        return {"naive_bayes": float(nb_fake), "logistic": float(lr_fake),
                "svm_margin": svm_margin, "known_words": int((x > 0).sum())}


def verdict(s):
    return "SUSPICIOUS style" if s["svm_margin"] > 0 else "NORMAL style"


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or input("Paste text: ")
    for name in ("isot", "liar"):
        if not (MODELS / name / "model.npz").exists():
            continue
        s = StyleModel(name).score(text)
        print(f"\n[{name.upper()} model]  {verdict(s)}")
        print(f"  Naive Bayes fake-score : {s['naive_bayes']:.2f}")
        print(f"  Logistic fake-score    : {s['logistic']:.2f}")
        print(f"  SVM margin             : {s['svm_margin']:+.2f}  (>0 = suspicious)")
        print(f"  Known words in text    : {s['known_words']}")
    print("\nNOTE: this checks writing style only. It does NOT verify facts.")