from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .rules import FREE_FORM_ROLES, _name_tokens, clean_text, make_dataset_finding, make_finding, numeric_share

PLACEHOLDER_WORDS = {
    "test", "tester", "dummy", "sample", "asdf", "asdfgh", "qwerty", "xxx", "xxxx", "foo", "bar",
    "lorem", "tbd", "todo", "n/a", "na", "none", "null", "unknown", "undefined", "placeholder",
}
FAKE_SEQUENCES = {"1234567", "12345678", "123456789", "1234567890", "0123456789", "12345", "123456", "qwerty", "asdfgh", "abcdef"}
FUTURE_DATE_TOKENS = {"expiry", "expire", "expires", "due", "end", "renewal", "deadline", "next", "scheduled", "delivery", "valid", "until"}
MIN_SIGNATURE_ROWS = 20


def _signature(text: pd.Series) -> pd.Series:
    sig = text.str.replace(r"[A-Za-z]", "A", regex=True).str.replace(r"\d", "9", regex=True)
    return sig.str.replace(r"(.)\1+", r"\1", regex=True)


def _date_checks(column: Any, text: pd.Series, n: int) -> List[Optional[Dict[str, Any]]]:
    parsed = pd.to_datetime(text, errors="coerce", format="mixed", utc=True)
    findings = [make_finding(
        "implausible_date", column, "high", (parsed < pd.Timestamp("1900-01-01", tz="UTC")).fillna(False).to_numpy(dtype=bool),
        f"Column '{column}' has dates before 1900", n,
    )]
    if not (_name_tokens(column)[0] & FUTURE_DATE_TOKENS):
        future = parsed > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
        findings.append(make_finding(
            "future_date", column, "high", future.fillna(False).to_numpy(dtype=bool),
            f"Column '{column}' has dates in the future", n,
        ))
    return findings


def _start_end_pairs(df: pd.DataFrame, roles: Dict[Any, Optional[str]], n: int) -> List[Optional[Dict[str, Any]]]:
    findings = []
    date_columns = [c for c in df.columns if roles.get(c) == "date"]
    for end in date_columns:
        tokens, _ = _name_tokens(end)
        if "end" not in tokens:
            continue
        for start in date_columns:
            if start != end and _name_tokens(start)[0] == (tokens - {"end"}) | {"start"}:
                first = pd.to_datetime(df[start], errors="coerce", format="mixed", utc=True)
                last = pd.to_datetime(df[end], errors="coerce", format="mixed", utc=True)
                findings.append(make_finding(
                    "end_before_start", end, "high", (last < first).fillna(False).to_numpy(dtype=bool),
                    f"Column '{end}' has dates earlier than '{start}'", n,
                ))
    return findings


def detect_errors(df: pd.DataFrame, roles: Dict[Any, Optional[str]]) -> List[Dict[str, Any]]:
    n = len(df)
    findings: List[Optional[Dict[str, Any]]] = []

    findings.append(make_finding("duplicate_rows", None, "medium", df.duplicated().to_numpy(), "Dataset contains duplicate rows", n))
    findings.extend(_start_end_pairs(df, roles, n))

    for column in df.columns:
        role = roles.get(column)
        text = clean_text(df[column])
        present = text.notna().to_numpy()

        if not present.any():
            findings.append(make_dataset_finding("empty_column", column, "high", f"Column '{column}' contains no values"))
            continue

        if role == "date":
            findings.extend(_date_checks(column, text, n))
        if role == "identifier":
            findings.append(make_finding(
                "duplicate_identifier", column, "high", text.duplicated(keep=False).to_numpy() & present,
                f"Column '{column}' has repeated identifiers", n,
            ))

        numeric, share = numeric_share(df[column])
        if share >= 0.8 and role not in FREE_FORM_ROLES:
            findings.append(make_finding("infinite_value", column, "high", np.isinf(numeric).to_numpy(), f"Column '{column}' has infinite values", n))
            finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite) >= 10:
                median = finite.median()
                mad = (finite - median).abs().median() * 1.4826
                if mad > 0:
                    extreme = ((numeric - median).abs() / mad > 10).fillna(False).to_numpy(dtype=bool)
                    findings.append(make_finding("extreme_value", column, "medium", extreme, f"Column '{column}' has values far beyond its typical spread", n))
            continue

        if role == "email":
            continue

        lowered = text.str.lower()
        alnum = lowered.fillna("").str.replace(r"[^a-z0-9]", "", regex=True)
        junk = (alnum.str.fullmatch(r"(.)\1{3,}").fillna(False) | alnum.isin(FAKE_SEQUENCES)).to_numpy(dtype=bool) & present
        if role in (None, "person_name"):
            junk |= lowered.isin(PLACEHOLDER_WORDS).fillna(False).to_numpy(dtype=bool)
        findings.append(make_finding("suspicious_pattern", column, "low", junk, f"Column '{column}' has placeholder or fake-looking values", n))

        if present.sum() >= MIN_SIGNATURE_ROWS:
            signature = _signature(text.dropna())
            shares = signature.value_counts(normalize=True)
            if shares.iloc[0] >= 0.85:
                rare = np.zeros(n, dtype=bool)
                rare[np.flatnonzero(present)] = (signature.map(shares) <= 0.05).to_numpy(dtype=bool)
                findings.append(make_finding("format_outlier", column, "medium", rare, f"Column '{column}' has values whose format differs from the dominant pattern", n))

    return [f for f in findings if f]
