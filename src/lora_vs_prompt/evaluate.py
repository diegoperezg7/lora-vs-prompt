"""Aggregate cached predictions into metrics with bootstrap intervals and cost estimates."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from . import config
from .data import Doc
from .inference import read_meta, read_predictions, select_docs
from .metrics import bootstrap, confusion, macro_f1, paired_diff, score_doc
from .schema import FIELDS


def score_rows(name: str, split: str, docs: list[Doc]) -> list[dict]:
    preds = {r["id"]: r for r in read_predictions(name, split)}
    missing = [d.id for d in docs if d.id not in preds]
    if missing:
        raise RuntimeError(f"{name}/{split}: {len(missing)} predictions missing, run `make eval`")
    rows = []
    for d in docs:
        p = preds[d.id]
        rows.append(
            {
                "id": d.id,
                "layout": d.layout,
                "tags": d.tags,
                "gold_category": d.label["category"],
                **score_doc(p["output"], d.label),
                **{k: p[k] for k in ("latency_s", "prompt_tokens", "gen_tokens", "gen_tps")},
                "retried": p.get("retried", False),
            }
        )
    return rows


def _rate(xs) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")


def summarise(name: str, split: str, docs: list[Doc]) -> dict:
    rows = score_rows(name, split, docs)
    n = len(rows)
    gold = [r["gold_category"] for r in rows]
    pred = [r["pred_category"] for r in rows]
    valid = np.array([r["json_valid"] for r in rows])
    exact = np.array([r["exact"] for r in rows])
    cat_ok = np.array([r["category_ok"] for r in rows])
    fields = np.array([[r["fields"][f] for f in FIELDS] for r in rows])
    lat = np.array([r["latency_s"] for r in rows])
    seed, b = config.SEED, config.BOOTSTRAP_SAMPLES
    g_arr, p_arr = np.array(gold), np.array(pred)

    slices = defaultdict(list)
    for r in rows:
        for t in r["tags"] or ["clean"]:
            slices[t].append(r)
    by_layout = defaultdict(list)
    for r in rows:
        by_layout[r["layout"]].append(r)

    meta = read_meta(name, split)
    mean_lat = float(lat.mean())
    kwh_per_doc = mean_lat * config.LOCAL_WATTS / 3600 / 1000
    return {
        "contender": name,
        "split": split,
        "n": n,
        "json_valid": _rate(valid),
        "category_accuracy": _rate(cat_ok),
        "macro_f1": macro_f1(gold, pred),
        "exact_match": _rate(exact),
        "field_accuracy": {f: _rate(fields[:, i]) for i, f in enumerate(FIELDS)},
        "mean_field_accuracy": float(fields.mean()),
        "ci": {
            "macro_f1": bootstrap(lambda ix: macro_f1(list(g_arr[ix]), list(p_arr[ix])), n, b, seed),
            "exact_match": bootstrap(lambda ix: exact[ix].mean(), n, b, seed),
            "json_valid": bootstrap(lambda ix: valid[ix].mean(), n, b, seed),
            "mean_field_accuracy": bootstrap(lambda ix: fields[ix].mean(), n, b, seed),
        },
        "latency_p50_s": float(np.percentile(lat, 50)),
        "latency_p95_s": float(np.percentile(lat, 95)),
        "latency_mean_s": mean_lat,
        "prompt_tokens_mean": float(np.mean([r["prompt_tokens"] for r in rows])),
        "gen_tokens_mean": float(np.mean([r["gen_tokens"] for r in rows])),
        "gen_tps_median": float(np.median([r["gen_tps"] for r in rows])),
        "retried": int(sum(r["retried"] for r in rows)),
        "peak_memory_gb": meta.get("peak_memory_gb"),
        "load_seconds": meta.get("load_seconds"),
        "hours_per_1k_docs": mean_lat * 1000 / 3600,
        "kwh_per_1k_docs": kwh_per_doc * 1000,
        "eur_per_1k_docs_local": kwh_per_doc * 1000 * config.EUR_PER_KWH,
        "slices": {
            t: {
                "n": len(rs),
                "exact_match": _rate([r["exact"] for r in rs]),
                "category_accuracy": _rate([r["category_ok"] for r in rs]),
            }
            for t, rs in sorted(slices.items())
        },
        "layouts": {
            k: {
                "n": len(rs),
                "exact_match": _rate([r["exact"] for r in rs]),
                "category_accuracy": _rate([r["category_ok"] for r in rs]),
            }
            for k, rs in sorted(by_layout.items())
        },
        "confusion": confusion(gold, pred).tolist(),
        "errors": sorted({r["error"] for r in rows if r["error"]})[:10],
    }


def paired(splits: dict[str, list[Doc]], names: list[str]) -> list[dict]:
    """Differences on the documents both contenders were scored on (a subsample if either used one)."""
    out = []
    for a, b in config.PAIRS:
        if a not in names or b not in names:
            continue
        for split in ("test", "hard"):
            ids = {d.id for d in select_docs(config.BY_NAME[a], splits[split])}
            ids &= {d.id for d in select_docs(config.BY_NAME[b], splits[split])}
            docs = [d for d in splits[split] if d.id in ids]
            ra, rb = score_rows(a, split, docs), score_rows(b, split, docs)
            row = {"a": a, "b": b, "split": split, "n": len(docs)}
            for key in ("exact", "category_ok"):
                xa = np.array([r[key] for r in ra])
                xb = np.array([r[key] for r in rb])
                row[key] = paired_diff(xa, xb, config.BOOTSTRAP_SAMPLES, config.SEED)
            out.append(row)
    return out


def evaluate_all(splits: dict[str, list[Doc]], names: list[str]) -> dict:
    results = {"contenders": {}, "training": {}}
    for name in names:
        c = config.BY_NAME[name]
        results["contenders"][name] = {
            s: summarise(name, s, select_docs(c, docs)) for s, docs in splits.items() if s in c.splits
        }
    results["paired"] = paired(splits, names)
    for adapter in config.ADAPTERS:
        path = config.ADAPTER_DIR / adapter / "training.json"
        if path.exists():
            results["training"][adapter] = json.loads(path.read_text())
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    return results
