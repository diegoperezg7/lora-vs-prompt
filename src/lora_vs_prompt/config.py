"""Experiment configuration: contenders, LoRA hyperparameters, cost assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
LORA_DATA_DIR = ROOT / "data" / "lora"
PRED_DIR = ROOT / "predictions"
ADAPTER_DIR = ROOT / "adapters"
RESULTS_DIR = ROOT / "results"
TRAINING_DIR = RESULTS_DIR / "training"
FIGURES_DIR = ROOT / "docs" / "figures"
EXPORT_DIR = ROOT / "export" / "evalsmith"

SEED = 13
MAX_TOKENS = 200
FEW_SHOT_K = 4
BOOTSTRAP_SAMPLES = 2000

MODELS = {
    "0.5B": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "1.5B": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "3B": "mlx-community/Qwen2.5-3B-Instruct-4bit",
}


@dataclass(frozen=True)
class Contender:
    name: str
    model: str
    prompt: str  # "full" (guide + rules) or "short" (one line, LoRA only)
    shots: int = 0
    adapter: str | None = None
    retry_of: str | None = None  # re-asks only the invalid outputs of that contender
    sample: int | None = None  # seeded, category-stratified subsample per split, to save compute
    splits: tuple[str, ...] = ("dev", "test", "hard")

    @property
    def size(self) -> str:
        return next(k for k, v in MODELS.items() if v == self.model)


CONTENDERS = [
    Contender("qwen0.5b-fewshot", MODELS["0.5B"], "full", shots=FEW_SHOT_K),
    Contender("qwen1.5b-zeroshot", MODELS["1.5B"], "full"),
    Contender("qwen1.5b-fewshot", MODELS["1.5B"], "full", shots=FEW_SHOT_K),
    Contender(
        "qwen1.5b-fewshot-retry", MODELS["1.5B"], "full", shots=FEW_SHOT_K, retry_of="qwen1.5b-fewshot"
    ),
    Contender("qwen3b-fewshot", MODELS["3B"], "full", shots=FEW_SHOT_K, sample=100, splits=("test", "hard")),
    Contender("qwen0.5b-lora", MODELS["0.5B"], "short", adapter="qwen0.5b"),
    Contender("qwen1.5b-lora", MODELS["1.5B"], "short", adapter="qwen1.5b"),
]
BY_NAME = {c.name: c for c in CONTENDERS}
# Paired comparisons (same documents, paired bootstrap): each LoRA against its prompting rivals.
PAIRS = [
    ("qwen0.5b-lora", "qwen0.5b-fewshot"),
    ("qwen1.5b-lora", "qwen1.5b-fewshot"),
    ("qwen1.5b-lora", "qwen3b-fewshot"),
]

# mlx-lm LoRA settings shared by both adapters (QLoRA: the base stays in 4 bits).
LORA = {
    "fine_tune_type": "lora",
    "optimizer": "adam",
    "num_layers": 16,
    "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.05},
    "learning_rate": 1e-4,
    "batch_size": 4,
    "iters": 400,  # 2 epochs over 800 training documents
    "max_seq_length": 1024,
    "mask_prompt": True,
    "steps_per_report": 10,
    "steps_per_eval": 50,
    "val_batches": -1,
    "save_every": 50,  # every checkpoint is kept; the one with the lowest dev loss is used
    "grad_checkpoint": True,  # recompute activations: keeps the 1.5B run under ~5 GB peak
    "seed": SEED,
}
ADAPTERS = {"qwen0.5b": MODELS["0.5B"], "qwen1.5b": MODELS["1.5B"]}

# Cost assumptions. Local energy is an estimate, not a measurement: powermetrics needs
# root. Override these when you have a wattmeter reading for your machine.
LOCAL_WATTS = 20.0  # average system draw of an M4 MacBook Air while generating
EUR_PER_KWH = 0.18  # Spanish household tariff, order of magnitude
HOURLY_LABOUR_EUR = 25.0  # cost of one person-hour labelling documents
LABEL_SECONDS_PER_DOC = 45  # time to label one document by hand (category + 9 fields)
USD_TO_EUR = 0.92

# Public list prices of hosted models (USD per million tokens), used only as a reference
# point. Prices change; check the providers' pricing pages before quoting them.
API_PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
}
