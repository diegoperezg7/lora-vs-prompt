"""Export the test set as an evalsmith golden set plus one outputs file per contender.

evalsmith's ``prompts`` section compares a baseline and a candidate on golden cases
(``[{id, input, checks}]``) and fails when a case that passed on the baseline fails on
the candidate. Each case here carries two checks:

* ``json_schema``: the output parses as JSON and every scored field equals the label
  (``const``), except the vendor, whose spelling legitimately varies;
* ``contains``: the vendor name appears in the output (case-insensitive).

These checks are stricter than ``metrics.py`` (no OCR tolerance on the vendor), so the
pass rate evalsmith reports sits a few points under the exact match in the results.
The suite gates future changes against the adapter that would ship, ``qwen1.5b-lora``.
"""

from __future__ import annotations

import json
import math

from . import config
from .data import load_split
from .inference import read_predictions
from .schema import _JSON_BLOCK

YAML = """# evalsmith suite: gate a prompt or adapter change on the lora-vs-prompt test set.
# Paths are relative to this file. Put the new outputs in outputs/ and point candidate at them;
# any case that passed on the baseline and fails on the candidate fails the run.
name: lora-vs-prompt-test

prompts:
  golden: golden.json
  baseline: outputs/{baseline}.json
  candidate: outputs/{candidate}.json
  judge:
    enabled: false

gates:
  prompt_pass_rate: {gate}
"""


def golden_case(doc) -> dict:
    lab = doc.label
    props = {k: {"const": v} for k, v in lab.items() if k != "vendor"}
    return {
        "id": doc.id,
        "input": doc.text,
        "checks": [
            {
                "type": "json_schema",
                "schema": {"type": "object", "required": list(props), "properties": props},
            },
            {"type": "contains", "value": lab["vendor"]},
        ],
    }


def passes(doc, output: str) -> bool:
    """The two checks of ``golden_case``, evaluated here to set the gate."""
    try:
        got = json.loads(extracted(output))
    except json.JSONDecodeError:
        return False
    lab = doc.label
    fields_ok = isinstance(got, dict) and all(
        k in got and got[k] == v for k, v in lab.items() if k != "vendor"
    )
    return fields_ok and lab["vendor"].casefold() in extracted(output).casefold()


def extracted(output: str) -> str:
    """The JSON object the model produced, without any text around it."""
    m = _JSON_BLOCK.search(output)
    return m.group(0) if m else output


def export_evalsmith(
    results: dict, baseline: str = "qwen1.5b-lora", candidate: str = "qwen1.5b-lora"
) -> None:
    out = config.EXPORT_DIR
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    test = load_split(config.DATA_DIR, "test")
    (out / "golden.json").write_text(json.dumps([golden_case(d) for d in test], ensure_ascii=False, indent=1))
    for name in results["contenders"]:
        if config.BY_NAME[name].sample:  # a subsample cannot be gated against the full golden set
            continue
        preds = {r["id"]: extracted(r["output"]) for r in read_predictions(name, "test")}
        (out / "outputs" / f"{name}.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1))
    gate = 0.0
    if candidate in results["contenders"]:
        # Gate a little under the candidate's current pass rate so noise does not fail CI.
        outputs = {r["id"]: r["output"] for r in read_predictions(candidate, "test")}
        rate = sum(passes(d, outputs[d.id]) for d in test) / len(test)
        gate = max(0.0, math.floor((rate - 0.05) * 100) / 100)
    (out / "evalsmith.yaml").write_text(YAML.format(baseline=baseline, candidate=candidate, gate=gate))
