from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EMAIL_PATTERN = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
PHONE_PATTERN = r"^\+?[\d\s().-]{7,20}$"
POSTCODE_PATTERNS = {
    "us": r"\d{5}(?:-\d{4})?",
    "uk": r"[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}",
    "ca": r"[A-Z]\d[A-Z]\s?\d[A-Z]\d",
    "in": r"\d{6}",
}

# Order matters: the first role whose hint matches a column name wins.
NAME_HINTS = (
    ("email", ("email", "mail")),
    ("phone", ("phone", "mobile", "telephone", "tel", "fax")),
    ("postcode", ("zip", "zipcode", "postcode", "postal", "pincode")),
    ("date", ("date", "dob", "timestamp", "datetime")),
    ("identifier", ("id", "uuid", "guid")),
    ("person_name", ("name",)),
    ("numeric_amount", ("amount", "price", "cost", "salary", "revenue", "total", "fee", "age", "quantity", "qty", "score", "spent")),
)
PROFILE_SEMANTIC_MAP = {
    "email": "email",
    "phone": "phone",
    "name": "person_name",
    "date": "date",
    "identifier": "identifier",
}
NON_NEGATIVE_TOKENS = {"amount", "price", "cost", "salary", "revenue", "quantity", "qty", "count", "total", "fee", "spent"}
FREE_FORM_ROLES = {"email", "phone", "postcode", "date", "identifier", "person_name"}


def _name_tokens(name: Any) -> Tuple[set, str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name)).lower()
    return set(re.split(r"[^a-z0-9]+", text)) - {""}, text


def expected_type_from_name(name: Any) -> Optional[str]:
    tokens, text = _name_tokens(name)
    for role, hints in NAME_HINTS:
        for hint in hints:
            if hint in tokens or (len(hint) >= 5 and hint in text):
                return role
    return None


def column_roles(columns: List[Any], profile: Optional[Dict[str, Any]] = None) -> Dict[Any, Optional[str]]:
    semantic = (profile or {}).get("metadata", {}).get("semantic_types", {})
    return {
        column: expected_type_from_name(column) or PROFILE_SEMANTIC_MAP.get(semantic.get(column))
        for column in columns
    }


def to_float(value: Any, digits: int = 4) -> Optional[float]:
    value = float(value)
    return round(value, digits) if np.isfinite(value) else None


def make_finding(
    check: str,
    column: Optional[Any],
    severity: str,
    mask: Any,
    message: str,
    total_rows: int,
) -> Optional[Dict[str, Any]]:
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count == 0 or total_rows == 0:
        return None
    return {
        "check": check,
        "column": None if column is None else str(column),
        "severity": severity,
        "affected_rows": count,
        "affected_ratio": round(count / total_rows, 4),
        "example_rows": [int(i) for i in np.flatnonzero(mask)[:10]],
        "message": f"{message} ({count} of {total_rows} rows)",
        "_mask": mask,
    }


def make_dataset_finding(check: str, column: Optional[Any], severity: str, message: str) -> Dict[str, Any]:
    return {
        "check": check,
        "column": None if column is None else str(column),
        "severity": severity,
        "affected_rows": None,
        "affected_ratio": None,
        "example_rows": [],
        "message": message,
        "_mask": None,
    }


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA)


def _no_match(text: pd.Series, pattern: str) -> np.ndarray:
    present = text.notna().to_numpy()
    matched = text.fillna("").str.match(pattern).fillna(False).to_numpy(dtype=bool)
    return present & ~matched


def numeric_share(series: pd.Series) -> Tuple[pd.Series, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    present = int(series.notna().sum())
    return numeric, (int(numeric.notna().sum()) / present if present else 0.0)


def _range_for(column: Any, rules: Dict[str, Any]) -> Tuple[Optional[Tuple[Optional[float], Optional[float]]], str]:
    explicit = (rules.get("numeric_ranges") or {}).get(column)
    if explicit:
        return tuple(explicit), "high"
    tokens, _ = _name_tokens(column)
    if "age" in tokens:
        return (0, 120), "high"
    if tokens & {"percent", "percentage", "pct"}:
        return (0, 100), "high"
    if tokens & NON_NEGATIVE_TOKENS:
        return (0, None), "medium"
    return None, "medium"


def _categorical_findings(column: Any, text: pd.Series, n: int) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    positions = np.flatnonzero(text.notna().to_numpy())
    raw = pd.Series(text.to_numpy()[positions])
    if len(raw) < 4:
        return findings
    unique = raw.nunique()
    if unique > 50 or unique / len(raw) > 0.5:
        return findings

    key = raw.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    canonical = raw.groupby(key).agg(lambda values: values.value_counts().idxmax())
    variant = (raw != key.map(canonical)).to_numpy()
    mask = np.zeros(n, dtype=bool)
    mask[positions[variant]] = True
    finding = make_finding(
        "categorical_inconsistency", column, "medium", mask,
        f"Column '{column}' uses inconsistent spellings/casing for the same category", n,
    )
    if finding:
        findings.append(finding)

    counts = raw.value_counts()
    frequent = [value for value in counts[counts >= 3].index[:50]]
    rare = [value for value in counts[counts == 1].index[:50]]
    typos = {
        value
        for value in rare
        for common in frequent
        if value.lower() != common.lower() and SequenceMatcher(None, value.lower(), common.lower()).ratio() >= 0.85
    }
    if typos:
        mask = np.zeros(n, dtype=bool)
        mask[positions[raw.isin(typos).to_numpy()]] = True
        finding = make_finding(
            "possible_typo", column, "low", mask,
            f"Column '{column}' has rare values that closely resemble frequent categories", n,
        )
        if finding:
            findings.append(finding)
    return findings


def validate_rules(df: pd.DataFrame, roles: Dict[Any, Optional[str]], rules: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    rules = rules or {}
    n = len(df)
    findings: List[Dict[str, Any]] = []

    def add(finding: Optional[Dict[str, Any]]) -> None:
        if finding:
            findings.append(finding)

    countries = rules.get("postcode_countries") or list(POSTCODE_PATTERNS)
    postcode_pattern = "(?i)^(?:" + "|".join(POSTCODE_PATTERNS[country] for country in countries) + ")$"

    for column in df.columns:
        role = roles.get(column)
        text = clean_text(df[column])
        present = text.notna().to_numpy()

        if role == "email":
            add(make_finding("invalid_email", column, "high", _no_match(text, EMAIL_PATTERN), f"Column '{column}' has invalid email addresses", n))
        elif role == "phone":
            digits = text.fillna("").str.replace(r"\D", "", regex=True).str.len().to_numpy()
            bad = _no_match(text, PHONE_PATTERN) | (present & ((digits < 7) | (digits > 15)))
            add(make_finding("invalid_phone", column, "high", bad, f"Column '{column}' has invalid phone numbers", n))
        elif role == "postcode":
            add(make_finding("invalid_postcode", column, "medium", _no_match(text, postcode_pattern), f"Column '{column}' has invalid postcodes", n))
        elif role == "date":
            parsed = pd.to_datetime(text, errors="coerce", format="mixed", utc=True)
            add(make_finding("invalid_date", column, "high", present & parsed.isna().to_numpy(), f"Column '{column}' has unparseable dates", n))

        allowed = (rules.get("allowed_values") or {}).get(column)
        if allowed is not None:
            bad = present & ~text.isin([str(value) for value in allowed]).fillna(False).to_numpy(dtype=bool)
            add(make_finding("value_not_allowed", column, "high", bad, f"Column '{column}' has values outside the allowed set", n))

        if role in FREE_FORM_ROLES:
            continue

        numeric, share = numeric_share(df[column])
        if share >= 0.8:
            bounds, severity = _range_for(column, rules)
            if bounds:
                low, high = bounds
                bad = np.zeros(n, dtype=bool)
                if low is not None:
                    bad |= (numeric < low).fillna(False).to_numpy(dtype=bool)
                if high is not None:
                    bad |= (numeric > high).fillna(False).to_numpy(dtype=bool)
                add(make_finding("numeric_out_of_range", column, severity, bad, f"Column '{column}' has values outside {low}..{high}", n))
        else:
            findings.extend(_categorical_findings(column, text, n))

    return findings
