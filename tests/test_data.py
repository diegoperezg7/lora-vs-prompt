import re

import pytest

from lora_vs_prompt.data import (
    CONCEPTS,
    DISTRACTORS,
    INJECTIONS,
    LAYOUTS,
    OWNER_HOME_CONCEPTS,
    RECHARGE_NO,
    RECHARGE_YES,
    VENDORS,
    generate,
    make_doc,
    ngram_overlap,
    stratified_sample,
)
from lora_vs_prompt.schema import Expense


@pytest.fixture(scope="module")
def splits():
    return generate(seed=13, sizes={"train": 300, "dev": 60, "test": 120, "hard": 80})


def test_generation_is_deterministic():
    a = generate(seed=7, sizes={"train": 20, "dev": 5, "test": 10})
    b = generate(seed=7, sizes={"train": 20, "dev": 5, "test": 10})
    assert [d.text for s in a.values() for d in s] == [d.text for s in b.values() for d in s]


def test_different_seed_changes_data():
    assert make_doc(1, "train", 0).text != make_doc(2, "train", 0).text


def test_labels_validate_against_schema(splits):
    for docs in splits.values():
        for d in docs:
            Expense.model_validate(d.label)


def test_no_text_leaks_between_splits(splits):
    texts = {s: {d.text for d in docs} for s, docs in splits.items()}
    names = list(texts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not texts[a] & texts[b], (a, b)


def test_test_layouts_and_vendors_are_held_out(splits):
    train_layouts = {d.layout for d in splits["train"]} | {d.layout for d in splits["dev"]}
    test_layouts = {d.layout for d in splits["test"]}
    assert train_layouts <= set(LAYOUTS["A"])
    assert test_layouts <= set(LAYOUTS["B"])
    assert not train_layouts & test_layouts
    train_vendors = {d.label["vendor"] for d in splits["train"]}
    test_vendors = {d.label["vendor"] for d in splits["test"]}
    assert not train_vendors & test_vendors


def test_vendor_pools_are_disjoint():
    a = {v for vs in VENDORS["A"].values() for v, _ in vs}
    b = {v for vs in VENDORS["B"].values() for v, _ in vs}
    assert not a & b


def test_amounts_are_consistent(splits):
    for docs in splits.values():
        for d in docs:
            lab = d.label
            if lab["category"] not in ("insurance", "professional_services"):
                assert abs(lab["base"] + lab["vat"] - lab["total"]) < 0.011, d.id
            if lab["period_start"]:
                assert lab["period_start"] <= lab["period_end"]


def test_total_is_printed_in_some_format(splits):
    """The label must be recoverable from the text: the total's digits appear."""
    for d in splits["train"][:200]:
        whole, cents = f"{d.label['total']:.2f}".split(".")
        digits = re.sub(r"\D", "", d.text)
        assert f"{whole}{cents}" in digits, d.id


def test_every_category_appears_in_every_split(splits):
    for docs in splits.values():
        assert len({d.label["category"] for d in docs}) == 9


def test_hard_cases_are_present(splits):
    tags = {t for d in splits["test"] for t in d.tags}
    assert {"injection", "owner_home", "repair_vs_improvement", "distractor_amount"} <= tags


def test_hard_split_uses_its_own_layouts_and_heavy_noise(splits):
    hard = splits["hard"]
    assert {d.layout for d in hard} <= set(LAYOUTS["C"])
    assert all("ocr_heavy" in d.tags for d in hard)
    train_vendors = {d.label["vendor"] for d in splits["train"]}
    assert not train_vendors & {d.label["vendor"] for d in hard}


def test_wording_pools_are_disjoint():
    for a_pool, b_pool in [
        (
            {c for cs in CONCEPTS["A"].values() for c, _ in cs},
            {c for cs in CONCEPTS["B"].values() for c, _ in cs},
        ),
        ({c for _, c in OWNER_HOME_CONCEPTS["A"]}, {c for _, c in OWNER_HOME_CONCEPTS["B"]}),
        (set(RECHARGE_YES["A"]), set(RECHARGE_YES["B"])),
        (set(RECHARGE_NO["A"]), set(RECHARGE_NO["B"])),
        (set(INJECTIONS["A"]), set(INJECTIONS["B"])),
        (set(DISTRACTORS["A"]), set(DISTRACTORS["B"])),
    ]:
        assert not a_pool & b_pool


def test_ngram_overlap_separates_seen_from_unseen(splits):
    dev = ngram_overlap(splits["train"], splits["dev"])
    test = ngram_overlap(splits["train"], splits["test"])
    hard = ngram_overlap(splits["train"], splits["hard"])
    assert dev["mean"] > 5 * test["mean"]
    assert test["mean"] < 0.05 and hard["mean"] < 0.05


def test_stratified_sample_keeps_category_mix(splits):
    docs = splits["test"]
    a = stratified_sample(docs, 50, seed=13)
    assert len(a) == 50 and a == stratified_sample(docs, 50, seed=13)
    full = {
        c: sum(d.label["category"] == c for d in docs) / len(docs)
        for c in {d.label["category"] for d in docs}
    }
    for cat, share in full.items():
        assert abs(sum(d.label["category"] == cat for d in a) - 50 * share) < 1
