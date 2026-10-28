"""Command line: lvp data | train | eval | report."""

from __future__ import annotations

import json
from collections import Counter

import typer

from . import config
from .data import generate, load_split, ngram_overlap, write_splits

app = typer.Typer(add_completion=False, no_args_is_help=True, help="When does LoRA beat prompting?")

SPLITS = ("dev", "test", "hard")


def _names(contender: str) -> list[str]:
    if contender == "all":
        return [c.name for c in config.CONTENDERS]
    names = [n.strip() for n in contender.split(",")]
    unknown = [n for n in names if n not in config.BY_NAME]
    if unknown:
        raise typer.BadParameter(f"unknown contender(s): {', '.join(unknown)}")
    return names


@app.command()
def data(seed: int = config.SEED) -> None:
    """Generate train/dev/test and the mlx-lm training files."""
    from .train import write_lora_data

    splits = generate(seed)
    write_splits(splits, config.DATA_DIR)
    write_lora_data(splits["train"], splits["dev"])
    stats = {
        s: {
            "docs": len(docs),
            "layouts": dict(Counter(d.layout for d in docs)),
            "categories": dict(Counter(d.label["category"] for d in docs)),
            "tags": dict(Counter(t for d in docs for t in d.tags)),
        }
        for s, docs in splits.items()
    }
    for s in ("dev", "test", "hard"):
        stats[s]["ngram_overlap_with_train"] = ngram_overlap(splits["train"], splits[s])
    (config.DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    for s, docs in splits.items():
        overlap = stats[s].get("ngram_overlap_with_train", {}).get("mean")
        extra = f"  6-gram overlap with train {overlap:.1%}" if overlap is not None else ""
        typer.echo(f"{s:<5} {len(docs):>4} docs  layouts: {', '.join(sorted(stats[s]['layouts']))}{extra}")


@app.command()
def train(adapter: str = typer.Option("all", help="qwen0.5b, qwen1.5b or all")) -> None:
    """Fine-tune the LoRA adapters (skips adapters that already exist unless --force)."""
    from .train import train_adapter

    names = list(config.ADAPTERS) if adapter == "all" else [adapter]
    for name in names:
        if (config.ADAPTER_DIR / name / "training.json").exists():
            typer.echo(f"{name}: adapter exists, delete adapters/{name} to retrain")
            continue
        typer.echo(f"{name}: training on {config.ADAPTERS[name]}")
        s = train_adapter(name, log=typer.echo)
        typer.echo(
            f"{name}: {s['train_seconds']:.0f}s, peak {s['peak_memory_gb']:.2f} GB, {s['adapter_mb']} MB"
        )


@app.command(name="eval")
def evaluate(
    contender: str = typer.Option("all", help="comma-separated contender names or all"),
    split: str = typer.Option("dev,test,hard"),
) -> None:
    """Run (or reuse cached) predictions, then score them into results/metrics.json."""
    from .evaluate import evaluate_all
    from .inference import run, select_docs

    train_docs = load_split(config.DATA_DIR, "train")
    splits = {s: load_split(config.DATA_DIR, s) for s in split.split(",")}
    names = _names(contender)
    for name in names:
        c = config.BY_NAME[name]
        for s, docs in splits.items():
            if s in c.splits:
                run(c, s, select_docs(c, docs), train_docs, log=typer.echo)
    all_splits = {s: load_split(config.DATA_DIR, s) for s in SPLITS}
    ready = [
        c.name
        for c in config.CONTENDERS
        if all((config.PRED_DIR / c.name / f"{s}.meta.json").exists() for s in c.splits)
    ]
    results = evaluate_all(all_splits, ready)
    for name in ready:
        t = results["contenders"][name]["test"]
        typer.echo(
            f"{name:<24} test exact {t['exact_match']:.3f}  macro-F1 {t['macro_f1']:.3f}  "
            f"valid {t['json_valid']:.3f}  p50 {t['latency_p50_s']:.2f}s"
        )


@app.command()
def report() -> None:
    """Tables, figures and the evalsmith export from results/metrics.json."""
    from .export import export_evalsmith
    from .report import build_report

    results = json.loads((config.RESULTS_DIR / "metrics.json").read_text())
    build_report(results)
    export_evalsmith(results)
    typer.echo(f"wrote {config.RESULTS_DIR / 'results.md'}, figures in {config.FIGURES_DIR}")


if __name__ == "__main__":
    app()
