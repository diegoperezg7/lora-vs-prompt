"""Run a contender over a split with mlx-lm and cache every prediction on disk.

Predictions are appended to ``predictions/<contender>/<split>.jsonl`` one line per
document, so an interrupted run resumes where it stopped. A fingerprint of everything
that changes the output (model, prompt, shots, adapter weights, decoding) is stored in
``<split>.meta.json``; a mismatch discards the stale cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

from . import config
from .compute import compute_turn
from .data import Doc, stratified_sample
from .prompts import RETRY_MESSAGE, SYSTEM_FULL, SYSTEM_SHORT, build_messages, pick_few_shot
from .schema import parse_output


def _adapter_path(c: config.Contender) -> Path | None:
    return config.ADAPTER_DIR / c.adapter if c.adapter else None


def fingerprint(c: config.Contender, shots: list[Doc]) -> str:
    h = hashlib.sha256()
    parts = [c.model, c.prompt, SYSTEM_FULL if c.prompt == "full" else SYSTEM_SHORT, str(config.MAX_TOKENS)]
    parts += [d.text for d in shots]
    if c.retry_of:
        parts += ["retry", RETRY_MESSAGE, c.retry_of]
    h.update("\x1f".join(parts).encode())
    if (ap := _adapter_path(c)) is not None:
        h.update((ap / "adapters.safetensors").read_bytes())
    return h.hexdigest()[:16]


def select_docs(c: config.Contender, docs: list[Doc]) -> list[Doc]:
    """The documents a contender is scored on: all of them, or its stratified subsample."""
    return stratified_sample(docs, c.sample, config.SEED) if c.sample else docs


def read_predictions(name: str, split: str) -> list[dict]:
    path = config.PRED_DIR / name / f"{split}.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_meta(name: str, split: str) -> dict:
    path = config.PRED_DIR / name / f"{split}.meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


class Runner:
    """Loads a model once and generates greedily."""

    def __init__(self, model_id: str, adapter: Path | None):
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        self.model, self.tok = load(model_id, adapter_path=str(adapter) if adapter else None)
        self.load_seconds = time.perf_counter() - t0
        self.generate([{"role": "user", "content": "Hola"}])  # warm-up, not timed

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        from mlx_lm import stream_generate

        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
        t0 = time.perf_counter()
        text, last = "", None
        for resp in stream_generate(self.model, self.tok, ids, max_tokens=config.MAX_TOKENS):
            text += resp.text
            last = resp
        latency = time.perf_counter() - t0
        return text, {
            "latency_s": round(latency, 4),
            "prompt_tokens": last.prompt_tokens,
            "gen_tokens": last.generation_tokens,
            "prompt_tps": round(last.prompt_tps, 1),
            "gen_tps": round(last.generation_tps, 1),
        }

    def peak_gb(self) -> float:
        return round(self.mx.get_peak_memory() / 1e9, 3)


def _prepare(c: config.Contender, split: str, shots: list[Doc]) -> tuple[Path, set[str], str]:
    out_dir = config.PRED_DIR / c.name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.jsonl"
    fp = fingerprint(c, shots)
    if read_meta(c.name, split).get("fingerprint") != fp and path.exists():
        path.unlink()
    done = {r["id"] for r in read_predictions(c.name, split)}
    return path, done, fp


def _write_meta(c: config.Contender, split: str, fp: str, runner: Runner | None, n: int) -> None:
    meta = read_meta(c.name, split)
    meta.update({"fingerprint": fp, "model": c.model, "adapter": c.adapter, "docs": n})
    if runner is not None:
        meta.update({"load_seconds": round(runner.load_seconds, 2), "peak_memory_gb": runner.peak_gb()})
    (config.PRED_DIR / c.name / f"{split}.meta.json").write_text(json.dumps(meta, indent=2))


def run(c: config.Contender, split: str, docs: list[Doc], train: list[Doc], log=print) -> None:
    if c.retry_of:
        return _run_retry(c, split, docs, train, log)
    ap = _adapter_path(c)
    if ap is not None and not (ap / "adapters.safetensors").exists():
        cached = {r["id"] for r in read_predictions(c.name, split)}
        if all(d.id in cached for d in docs):  # fresh clone: trust the committed predictions
            log(f"{c.name}/{split}: cached ({len(cached)} docs, adapter not present locally)")
            return
        raise FileNotFoundError(f"{ap} is missing: run `make train` first")
    shots = pick_few_shot(train, c.shots) if c.shots else []
    path, done, fp = _prepare(c, split, shots)
    todo = [d for d in docs if d.id not in done]
    if not todo:
        log(f"{c.name}/{split}: cached ({len(done)} docs)")
        return
    log(f"{c.name}/{split}: {len(todo)} to run, {len(done)} cached")
    with compute_turn():
        runner = Runner(c.model, _adapter_path(c))
        with path.open("a", encoding="utf-8") as fh:
            for i, doc in enumerate(todo, 1):
                text, stats = runner.generate(build_messages(doc.text, c.prompt, shots))
                fh.write(json.dumps({"id": doc.id, "output": text, **stats}, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 25 == 0:
                    log(f"  {i}/{len(todo)}  last {stats['latency_s']:.2f}s")
        _write_meta(c, split, fp, runner, len(docs))
        del runner


def _run_retry(c: config.Contender, split: str, docs: list[Doc], train: list[Doc], log) -> None:
    """Re-ask once, with the validation error, only where the base contender failed."""
    base = config.BY_NAME[c.retry_of]
    base_preds = {r["id"]: r for r in read_predictions(base.name, split)}
    if len(base_preds) < len(docs):
        raise RuntimeError(f"run {base.name} on {split} first")
    shots = pick_few_shot(train, c.shots) if c.shots else []
    path, done, fp = _prepare(c, split, shots)
    texts = {d.id: d.text for d in docs}
    todo = [d.id for d in docs if d.id not in done]
    failing = [i for i in todo if parse_output(base_preds[i]["output"])[1] is None]
    log(f"{c.name}/{split}: {len(failing)} of {len(todo)} need a retry")
    runner = None
    with compute_turn() if failing else nullcontext():
        if failing:
            runner = Runner(c.model, _adapter_path(c))
        with path.open("a", encoding="utf-8") as fh:
            for doc_id in todo:
                rec = dict(base_preds[doc_id], retried=False)
                if doc_id in failing:
                    _, _, error = parse_output(rec["output"])
                    msgs = build_messages(texts[doc_id], c.prompt, shots)
                    msgs += [
                        {"role": "assistant", "content": rec["output"]},
                        {"role": "user", "content": RETRY_MESSAGE.format(error=error)},
                    ]
                    text, stats = runner.generate(msgs)
                    rec = {
                        "id": doc_id,
                        "output": text,
                        "retried": True,
                        "latency_s": round(rec["latency_s"] + stats["latency_s"], 4),
                        "prompt_tokens": rec["prompt_tokens"] + stats["prompt_tokens"],
                        "gen_tokens": rec["gen_tokens"] + stats["gen_tokens"],
                        "prompt_tps": stats["prompt_tps"],
                        "gen_tps": stats["gen_tps"],
                    }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        base_meta = read_meta(base.name, split)
        _write_meta(c, split, fp, runner, len(docs))
        if runner is None:  # nothing re-run: inherit the base run's memory figure
            meta = read_meta(c.name, split)
            meta.update({k: base_meta[k] for k in ("load_seconds", "peak_memory_gb") if k in base_meta})
            (config.PRED_DIR / c.name / f"{split}.meta.json").write_text(json.dumps(meta, indent=2))
