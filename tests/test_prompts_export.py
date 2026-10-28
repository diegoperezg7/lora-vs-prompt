import json

from lora_vs_prompt.data import generate
from lora_vs_prompt.export import extracted, golden_case, passes
from lora_vs_prompt.prompts import SYSTEM_FULL, build_messages, pick_few_shot, target_json
from lora_vs_prompt.schema import CATEGORIES


def _splits():
    return generate(seed=13, sizes={"train": 300, "dev": 10, "test": 10})


def test_full_prompt_lists_every_category():
    for c in CATEGORIES:
        assert c in SYSTEM_FULL


def test_few_shot_examples_come_from_train_and_are_clean():
    s = _splits()
    shots = pick_few_shot(s["train"], 4)
    assert [d.label["category"] for d in shots] == [
        "insurance",
        "professional_services",
        "utilities",
        "improvement",
    ]
    assert all(d.split == "train" for d in shots)
    assert not any(t.startswith("ocr") or t == "injection" for d in shots for t in d.tags)


def test_message_shapes():
    s = _splits()
    doc = s["test"][0].text
    assert [m["role"] for m in build_messages(doc, "short")] == ["system", "user"]
    msgs = build_messages(doc, "full", pick_few_shot(s["train"], 2))
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert json.loads(msgs[2]["content"])["category"] == "insurance"


def test_golden_case_matches_evalsmith_shape():
    doc = _splits()["test"][0]
    case = golden_case(doc)
    assert set(case) == {"id", "input", "checks"}
    schema = case["checks"][0]["schema"]
    assert "vendor" not in schema["properties"]
    assert schema["properties"]["total"] == {"const": doc.label["total"]}
    assert case["checks"][1] == {"type": "contains", "value": doc.label["vendor"]}


def test_extracted_strips_surrounding_text():
    assert extracted('Sure!\n{"a": 1}\nThanks') == '{"a": 1}'
    assert extracted("no json") == "no json"


def test_passes_mirrors_the_golden_checks():
    doc = _splits()["test"][0]
    right = target_json(doc.label)
    assert passes(doc, "Here it is: " + right)
    wrong = json.loads(right) | {"total": doc.label["total"] + 1}
    assert not passes(doc, json.dumps(wrong, ensure_ascii=False))
    assert not passes(doc, "not json")
