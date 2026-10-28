import json

import numpy as np
import pytest

from lora_vs_prompt.metrics import (
    INVALID,
    bootstrap,
    confusion,
    field_correct,
    macro_f1,
    norm_amount,
    norm_date,
    norm_vendor,
    paired_diff,
    score_doc,
)
from lora_vs_prompt.schema import parse_output

GOLD = {
    "category": "utilities",
    "vendor": "Aguas del Litoral S.A.",
    "date": "2025-03-04",
    "base": 1234.5,
    "vat": 123.45,
    "total": 1357.95,
    "period_start": "2025-01-01",
    "period_end": "2025-02-28",
    "recharged_to_tenant": True,
}


def test_vendor_normalisation_ignores_case_accents_and_legal_suffix():
    assert norm_vendor("AGUAS DEL LITORAL, S.A.") == norm_vendor("Aguas del Litoral")
    assert norm_vendor("Eléctrica Peninsular S.L.") == "electrica peninsular"
    assert norm_vendor(None) == ""


@pytest.mark.parametrize(
    "raw, expected",
    [
        (1234.5, 1234.5),
        ("1.234,50", 1234.5),
        ("1234,50 €", 1234.5),
        ("1234.50", 1234.5),
        (True, None),
        ("x", None),
    ],
)
def test_amount_normalisation(raw, expected):
    assert norm_amount(raw) == expected


def test_date_normalisation():
    assert norm_date("04/03/2025") == "2025-03-04"
    assert norm_date("2025-03-04") == "2025-03-04"
    assert norm_date("4 de marzo") == "?"
    assert norm_date(None) is None


def test_field_correct_tolerates_a_cent_only():
    assert field_correct("total", 1357.96, 1357.95)
    assert not field_correct("total", 1357.98, 1357.95)
    assert not field_correct("recharged_to_tenant", "yes", True)


def test_perfect_output_scores_exact():
    s = score_doc(json.dumps(GOLD), GOLD)
    assert s["json_valid"] and s["exact"] and all(s["fields"].values())


def test_json_inside_prose_and_fences_is_recovered():
    s = score_doc("Here you go:\n```json\n" + json.dumps(GOLD) + "\n```", GOLD)
    assert s["json_valid"] and s["exact"]


def test_garbage_scores_nothing():
    s = score_doc("I cannot help with that", GOLD | {"period_start": None, "period_end": None})
    assert not s["json_valid"] and s["pred_category"] == INVALID
    assert not any(s["fields"].values())  # a missing period must not count as correct


def test_schema_violation_keeps_field_credit():
    bad = dict(GOLD, date="04/03/2025", extra="x")
    raw, valid, error = parse_output(json.dumps(bad))
    assert valid is None and error.startswith("schema")
    s = score_doc(json.dumps(bad), GOLD)
    assert not s["json_valid"] and s["fields"]["date"] and s["category_ok"]


def test_unknown_category_is_invalid():
    s = score_doc(json.dumps(dict(GOLD, category="gas")), GOLD)
    assert s["pred_category"] == INVALID and not s["category_ok"]


def test_macro_f1_hand_computed():
    gold = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", "b"]
    # a: tp1 fp0 fn1 -> 2/3 ; b: tp2 fp1 fn0 -> 4/5
    assert macro_f1(gold, pred, labels=["a", "b"]) == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_confusion_has_invalid_column():
    m = confusion(["utilities", "utilities"], ["utilities", INVALID])
    assert m.shape == (9, 10) and m.sum() == 2 and m[1, -1] == 1


def test_bootstrap_interval_contains_mean_and_is_seeded():
    x = np.array([1] * 70 + [0] * 30)
    lo, hi = bootstrap(lambda ix: x[ix].mean(), len(x), 500, seed=1)
    assert lo < 0.7 < hi
    assert (lo, hi) == bootstrap(lambda ix: x[ix].mean(), len(x), 500, seed=1)


def test_paired_diff_detects_a_clear_gap():
    a = np.array([1] * 90 + [0] * 10)
    b = np.array([1] * 50 + [0] * 50)
    mean, lo, hi = paired_diff(a, b, 500, seed=0)
    assert mean == pytest.approx(0.4) and lo > 0


def test_vendor_tolerates_ocr_noise_but_not_a_different_name():
    gold = "Fontanería Martínez S.L."
    assert field_correct("vendor", "F0ntanería Martínez S.L.", gold)
    assert field_correct("vendor", "Fontaneria Martlnez", gold)
    assert not field_correct("vendor", "Electricidad Martínez", gold)
    assert not field_correct(
        "vendor",
        "Administración de Fincas Lafuente",
        "Administración de Fincas Lafuente (C.P. Paseo Sagasta 22)",
    )
