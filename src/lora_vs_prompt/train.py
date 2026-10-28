"""LoRA fine-tuning with mlx-lm, logging loss curves and wall time to the adapter folder."""

from __future__ import annotations

import json
import shutil
import time
import types

from . import config
from .compute import compute_turn
from .data import Doc
from .prompts import build_messages, target_json


def write_lora_data(train: list[Doc], dev: list[Doc]) -> None:
    """mlx-lm chat format; ``valid`` is the dev split (seen layouts, unseen documents)."""
    config.LORA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, docs in (("train", train), ("valid", dev)):
        with (config.LORA_DATA_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for d in docs:
                msgs = build_messages(d.text, "short")
                msgs.append({"role": "assistant", "content": target_json(d.label)})
                fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")


def train_adapter(name: str, log=print) -> dict:
    import mlx.core as mx
    import numpy as np
    from mlx_lm.lora import CONFIG_DEFAULTS, load_dataset, train_model
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.utils import load

    out = config.ADAPTER_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    args = types.SimpleNamespace(
        **{
            **CONFIG_DEFAULTS,
            **config.LORA,
            "model": config.ADAPTERS[name],
            "data": str(config.LORA_DATA_DIR),
            "train": True,
            "test": False,
            "adapter_path": str(out),
            "resume_adapter_file": None,
            "report_to": None,
            "project_name": None,
            "trust_remote_code": False,
        }
    )
    curves: dict[str, list] = {"train": [], "val": []}

    class Recorder(TrainingCallback):
        def on_train_loss_report(self, info: dict) -> None:
            curves["train"].append(info)

        def on_val_loss_report(self, info: dict) -> None:
            curves["val"].append(info)
            log(f"  iter {info['iteration']:>4}  val loss {info['val_loss']:.4f}")

    with compute_turn():
        np.random.seed(args.seed)
        mx.random.seed(args.seed)
        mx.reset_peak_memory()
        model, tokenizer = load(args.model)
        train_set, valid_set, _ = load_dataset(args, tokenizer)
        t0 = time.perf_counter()
        train_model(args, model, train_set, valid_set, Recorder())
        seconds = time.perf_counter() - t0
        peak = mx.get_peak_memory() / 1e9

    # Keep the checkpoint with the lowest dev loss. Validation at iteration k runs before
    # update k + 1, so it scores the weights saved as checkpoint k + 1 minus one step.
    saved = {int(p.name.split("_")[0]): p for p in out.glob("*_adapters.safetensors")}
    best = min(curves["val"], key=lambda v: v["val_loss"])
    best_ckpt = min(saved, key=lambda it: abs(it - (best["iteration"] + 1)))
    shutil.copyfile(saved[best_ckpt], out / "adapters.safetensors")

    summary = {
        "adapter": name,
        "selected_checkpoint": best_ckpt,
        "selected_val_loss": best["val_loss"],
        "model": args.model,
        "train_seconds": round(seconds, 1),
        "peak_memory_gb": round(peak, 3),
        "trained_tokens": curves["train"][-1]["trained_tokens"] if curves["train"] else 0,
        "adapter_mb": round((out / "adapters.safetensors").stat().st_size / 1e6, 2),
        "hyperparameters": config.LORA,
        "curves": curves,
    }
    text = json.dumps(summary, indent=2, default=float)
    (out / "training.json").write_text(text)
    # adapters/ is not in git; the copy under results/ keeps curves and timings with the repo.
    config.TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    (config.TRAINING_DIR / f"{name}.json").write_text(text)
    return summary
