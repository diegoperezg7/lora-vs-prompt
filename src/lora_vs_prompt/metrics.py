"""Scoring: field normalisation, macro-F1, confusion matrix, bootstrap intervals."""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np

from .schema import CATEGORIES, FIELDS, parse_output

INVALID = "invalid"
_LEGAL_SUFFIXES = {"sl", "sa", "slu", "sau", "cb", "scp"}
_DATE_PATTERNS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y")


VENDOR_SIMILARITY = 0.85


def norm_vendor(value) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(t for t in text.split() if t not in _LEGAL_SUFFIXES)


def norm_amount(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return round(float(value), 2)
    if isinstance(value, str):
        s = re.sub(r"[^\d,.\-]", "", value)
        if "," in s:  # Spanish format: dots group thousands, comma is the decimal mark
            s = s.replace(".", "").replace(",", ".")
        try:
            return round(float(s), 2)
        except ValueError:
            return None
    return None


def norm_date(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return "?"
    for pattern in _DATE_PATTERNS:
        try:
            return dt.datetime.strptime(value.strip(), pattern).date().isoformat()
        except ValueError:
            continue
    return "?"


def norm_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def field_correct(name: str, pred, gold) -> bool:
    if name == "vendor":
        p, g = norm_vendor(pred), norm_vendor(gold)
        # OCR noise also hits the printed vendor name (o->0, m->rn), so an exact copy of the
        # document can differ from the clean label; near-identical strings count as right.
        return bool(p) and SequenceMatcher(None, p, g).ratio() >= VENDOR_SIMILARITY
    if name in ("base", "vat", "total"):
        p = norm_amount(pred)
        return p is not None and abs(p - gold) <= 0.011
    if name in ("date", "period_start", "period_end"):
        return norm_date(pred) == gold
    if name == "recharged_to_tenant":
        return norm_bool(pred) is gold
    raise KeyError(name)


def score_doc(output: str, gold: dict) -> dict:
    """Per-document result: validity, predicted category and one boolean per field."""
    raw, valid, error = parse_output(output)
    raw = raw or {}
    category = raw.get("category") if raw.get("category") in CATEGORIES else INVALID
    # A missing period key means "no period"; any other missing field is wrong.
    fields = {
        f: bool(raw) and (f in raw or f.startswith("period")) and field_correct(f, raw.get(f), gold[f])
        for f in FIELDS
    }
    return {
        "json_valid": valid is not None,
        "error": error,
        "pred_category": category,
        "category_ok": category == gold["category"],
        "fields": fields,
        "exact": category == gold["category"] and all(fields.values()),
    }


def macro_f1(gold: list[str], pred: list[str], labels: list[str] = CATEGORIES) -> float:
    g, p = np.asarray(gold), np.asarray(pred)
    f1s = []
    for lab in labels:
        tp = np.sum((g == lab) & (p == lab))
        fp = np.sum((g != lab) & (p == lab))
        fn = np.sum((g == lab) & (p != lab))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def confusion(gold: list[str], pred: list[str]) -> np.ndarray:
    cols = CATEGORIES + [INVALID]
    m = np.zeros((len(CATEGORIES), len(cols)), dtype=int)
    for g, p in zip(gold, pred, strict=True):
        m[CATEGORIES.index(g), cols.index(p)] += 1
    return m


def bootstrap(stat, n: int, samples: int, seed: int) -> tuple[float, float]:
    """Percentile 95% interval of ``stat(indices)`` over document resamples."""
    rng = np.random.default_rng(seed)
    values = [stat(rng.integers(0, n, n)) for _ in range(samples)]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def paired_diff(a: np.ndarray, b: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    """Mean of a - b with a paired bootstrap interval (same resampled documents for both)."""
    rng = np.random.default_rng(seed)
    d = a.astype(float) - b.astype(float)
    values = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(samples)]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
