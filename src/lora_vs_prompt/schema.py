"""Output schema shared by the generator, the prompts and the evaluator."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Category = Literal[
    "repairs_maintenance",
    "utilities",
    "community_fees",
    "property_tax",
    "insurance",
    "mortgage_interest",
    "improvement",
    "professional_services",
    "non_deductible",
]

CATEGORIES: list[str] = list(Category.__args__)

# One line per category, used verbatim in the prompts.
CATEGORY_GUIDE: dict[str, str] = {
    "repairs_maintenance": (
        "repairs and upkeep that keep the property as it was"
        " (plumber, painter, locksmith, like-for-like replacements)"
    ),
    "utilities": "electricity, water, gas, internet and similar supplies of the rented property",
    "community_fees": "homeowners' association (comunidad de propietarios) fees and levies",
    "property_tax": "IBI and other local taxes on the property (rubbish collection, vado)",
    "insurance": "home, liability or rent-default insurance premiums for the property",
    "mortgage_interest": "loan or mortgage instalments and interest for the property",
    "improvement": (
        "works that add value, capacity or new elements"
        " (renovations, extensions, new installations); amortised, not expensed"
    ),
    "professional_services": "letting agency, property manager, lawyer, tax advisor, notary",
    "non_deductible": (
        "anything not linked to the rented property:"
        " personal spending, the owner's own home, fines, penalties"
    ),
}


class Expense(BaseModel):
    """Target JSON for one expense document."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    vendor: str = Field(description="Issuer name as printed, without the tax id")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Issue date, ISO 8601")
    base: float = Field(description="Net amount before VAT (taxable base, or net premium)")
    vat: float = Field(description="VAT (IVA) amount; 0 when exempt or not applicable")
    total: float = Field(description="Total amount charged on the document")
    period_start: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    period_end: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    recharged_to_tenant: bool = Field(description="True if the document says it is passed on to the tenant")


FIELDS: list[str] = [f for f in Expense.model_fields if f != "category"]

SCHEMA_HINT = (
    '{"category": one of ['
    + ", ".join(CATEGORIES)
    + '], "vendor": str, "date": "YYYY-MM-DD", "base": number, "vat": number, "total": number, '
    '"period_start": "YYYY-MM-DD" or null, "period_end": "YYYY-MM-DD" or null, '
    '"recharged_to_tenant": bool}'
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_output(text: str) -> tuple[dict | None, Expense | None, str | None]:
    """Parse raw model text.

    Returns (raw_dict, validated, error). ``raw_dict`` is whatever JSON object could be
    recovered, so field-level scoring still works when schema validation fails.
    """
    match = _JSON_BLOCK.search(text)
    if not match:
        return None, None, "no JSON object found"
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, None, f"invalid JSON: {exc.msg}"
    if not isinstance(raw, dict):
        return None, None, "JSON is not an object"
    try:
        return raw, Expense.model_validate(raw), None
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        return raw, None, f"schema: {loc}: {first['msg']}"
