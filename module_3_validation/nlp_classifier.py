from __future__ import annotations

import random
import re
import string
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .rules import clean_text, make_dataset_finding, make_finding

ROLE_TO_CLASS = {
    "email": "email",
    "phone": "phone",
    "postcode": "postcode",
    "date": "date",
    "identifier": "identifier",
    "person_name": "person_name",
    "numeric_amount": "numeric",
}
# Digit-only values are inherently ambiguous between these classes.
AMBIGUOUS = {"numeric", "postcode", "identifier"}
STRUCTURED_CLASSES = {"email", "phone", "postcode", "date", "identifier", "numeric"}
MAX_VALUES = 2000
MIN_CONFIDENCE = 0.5
MIN_DOMINANCE = 0.8
FLAG_PROBABILITY = 0.8

FIRST = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Irene", "Jack", "Karen", "Liam", "Mona", "Nathan",
    "Olivia", "Peter", "Quinn", "Rita", "Sam", "Tina", "Umar", "Vera", "Wei", "Xavier", "Yusuf", "Zara", "Aarav", "Priya",
    "Mohammed", "Fatima", "Chen", "Sofia", "Lucas", "Emma", "Noah", "Ava", "Mateo", "Isabella", "Hiroshi", "Yuki", "Ivan",
    "Natasha", "Kwame", "Amara", "Diego", "Carmen", "Lars", "Ingrid", "Omar", "Leila",
]
LAST = [
    "Johnson", "Smith", "Davis", "Lee", "Brown", "White", "Green", "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas",
    "Jackson", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson", "Clark", "Lewis", "Walker", "Hall", "Allen",
    "Young", "King", "Wright", "Patel", "Sharma", "Singh", "Khan", "Wang", "Zhang", "Kim", "Nguyen", "Tanaka", "Sato", "Ivanov",
    "Petrov", "Mensah", "Okafor", "Silva", "Santos", "Hansen", "Larsen", "Haddad", "Rahman", "Costa", "Fischer",
]
TITLES = ["Mr", "Mrs", "Ms", "Dr", "Prof"]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "example.com", "company.org", "mail.co.uk", "university.edu", "corp.net", "provider.com", "domain.net"]
CITIES = [
    "New York", "Chicago", "Los Angeles", "Seattle", "Miami", "Boston", "Houston", "Denver", "Austin", "Portland", "London",
    "Manchester", "Paris", "Berlin", "Madrid", "Rome", "Toronto", "Vancouver", "Sydney", "Melbourne", "Mumbai", "Delhi",
    "Bangalore", "Tokyo", "Osaka", "Singapore", "Dubai", "Cairo", "Lagos", "Nairobi", "Sao Paulo", "Mexico City", "Dublin",
    "San Diego", "Phoenix", "Atlanta", "Dallas", "Oslo", "Stockholm", "Vienna",
]
STATUS = [
    "active", "inactive", "pending", "shipped", "delivered", "cancelled", "returned", "open", "closed", "approved", "rejected",
    "Active", "Pending", "Shipped", "VIP", "Verified", "Premium", "Standard", "Basic", "Gold", "Silver", "High value", "Needs follow-up",
]
WORDS = [
    "customer", "order", "needs", "follow", "up", "active", "high", "value", "regular", "contacted", "email", "call", "back",
    "pending", "review", "account", "verified", "payment", "late", "premium", "support", "ticket", "resolved", "new", "returning",
    "prefers", "morning", "delivery", "discount", "applied", "note", "issue", "refund", "requested", "shipping", "address", "changed",
]
COUNTRIES = ["United States", "Canada", "United Kingdom", "Germany", "France", "India", "Japan", "Australia", "Brazil", "Mexico", "Spain", "Italy"]


def _email(r: random.Random) -> str:
    first, last = r.choice(FIRST).lower(), r.choice(LAST).lower()
    local = r.choice([f"{first}.{last}", f"{first}{last}", f"{first[0]}{last}", f"{first}_{last}{r.randint(1, 99)}", f"{first}{r.randint(1970, 2005)}"])
    value = f"{local}@{r.choice(DOMAINS)}"
    return value.upper() if r.random() < 0.08 else value


def _phone(r: random.Random) -> str:
    a, b, c = r.randint(200, 999), r.randint(200, 999), r.randint(0, 9999)
    return r.choice([
        f"{a}-{b}-{c:04d}", f"({a}) {b}-{c:04d}", f"+1 {a} {b} {c:04d}", f"+1-{a}-{b}-{c:04d}", f"{a}{b}{c:04d}", f"{a}.{b}.{c:04d}",
        f"+44 {r.randint(20, 79)} {r.randint(1000, 9999)} {r.randint(1000, 9999)}", f"+91 {r.randint(70000, 99999)} {r.randint(10000, 99999)}",
    ])


def _postcode(r: random.Random) -> str:
    up = string.ascii_uppercase
    return r.choice([
        f"{r.randint(10000, 99999)}", f"{r.randint(10000, 99999)}-{r.randint(1000, 9999)}",
        f"{r.choice(up)}{r.choice(up)}{r.randint(1, 9)} {r.randint(1, 9)}{r.choice(up)}{r.choice(up)}",
        f"{r.choice(up)}{r.randint(1, 9)}{r.choice(up)} {r.randint(1, 9)}{r.choice(up)}{r.randint(1, 9)}", f"{r.randint(100000, 999999)}",
    ])


def _date(r: random.Random) -> str:
    moment = datetime(1960, 1, 1) + timedelta(days=r.randint(0, 25000), seconds=r.randint(0, 86399))
    fmt = r.choice(["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d %b %Y", "%b %d, %Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"])
    return moment.strftime(fmt)


def _person_name(r: random.Random) -> str:
    first, last = r.choice(FIRST), r.choice(LAST)
    return r.choice([
        f"{first} {last}", f"{first} {last}", f"{last}, {first}", f"{first} {r.choice(string.ascii_uppercase)}. {last}",
        f"{r.choice(TITLES)} {first} {last}", f"{first.lower()} {last.lower()}", f"{first.upper()} {last.upper()}", first,
    ])


def _identifier(r: random.Random) -> str:
    up = string.ascii_uppercase
    uuid = "-".join("".join(r.choices("0123456789abcdef", k=size)) for size in (8, 4, 4, 4, 12))
    return r.choice([
        f"C-{r.randint(1000, 99999)}", f"ID{r.randint(1, 999999):06d}", uuid, f"ORD-{r.randint(2015, 2026)}-{r.randint(1, 99999):05d}",
        f"{r.choice(up)}{r.randint(100, 9999)}", f"{r.choice(up)}{r.choice(up)}-{r.randint(1000, 99999)}", "".join(r.choices(up + string.digits, k=r.randint(8, 12))),
    ])


def _numeric(r: random.Random) -> str:
    return r.choice([
        str(r.randint(0, 9999)), f"{r.uniform(0, 10000):.2f}", f"{r.uniform(-500, 500):.1f}", f"{r.randint(1000, 9999999):,}",
        f"${r.uniform(1, 5000):,.2f}", f"{r.randint(0, 100)}%", f"{r.uniform(0, 1):.4f}", f"{r.randint(18, 90)}.0", f"{r.uniform(0, 999):.3f}",
    ])


def _text(r: random.Random) -> str:
    return r.choice([
        r.choice(CITIES), r.choice(STATUS), " ".join(r.choices(WORDS, k=r.randint(2, 6))).capitalize(), r.choice(COUNTRIES),
        r.choice(CITIES), r.choice(STATUS),
    ])


GENERATORS = {
    "email": _email, "phone": _phone, "postcode": _postcode, "date": _date,
    "person_name": _person_name, "identifier": _identifier, "numeric": _numeric, "text": _text,
}


def _with_shape(value: str) -> str:
    # The shape (digits->9, letters->a/A) exposes formats that raw n-grams miss, e.g. 3-3-4 phone groups.
    shape = re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", value)))
    return f"{value}\x01{shape}"


@lru_cache(maxsize=1)
def get_classifier(seed: int = 7, per_class: int = 500) -> Pipeline:
    rng = random.Random(seed)
    values: List[str] = []
    labels: List[str] = []
    for label, generate in GENERATORS.items():
        for _ in range(per_class):
            values.append(generate(rng))
            labels.append(label)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 6), preprocessor=_with_shape, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=10.0, max_iter=2000, random_state=seed)),
    ])
    return model.fit(values, labels)


def _expected_class(role: Optional[str]) -> Optional[str]:
    return ROLE_TO_CLASS.get(role) if role else None


def classify_columns(df: pd.DataFrame, roles: Dict[Any, Optional[str]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Predict each column's meaning from its values; flag mislabeled columns and off-type values."""
    model = get_classifier()
    classes = [str(name) for name in model.classes_]
    n = len(df)
    classification: Dict[str, Any] = {}
    findings: List[Dict[str, Any]] = []

    for column in df.columns:
        text = clean_text(df[column])
        positions = np.flatnonzero(text.notna().to_numpy())[:MAX_VALUES]
        if len(positions) == 0:
            continue

        values = [str(v)[:100] for v in text.to_numpy()[positions]]
        probabilities = model.predict_proba(values)
        mean_probability = probabilities.mean(axis=0)
        predicted = classes[int(mean_probability.argmax())]
        confidence = float(mean_probability.max())
        per_value = np.array(classes)[probabilities.argmax(axis=1)]
        dominance = float((per_value == predicted).mean())
        expected = _expected_class(roles.get(column))

        status = "ok"
        if expected and predicted != expected and dominance >= MIN_DOMINANCE and confidence >= MIN_CONFIDENCE and not {expected, predicted} <= AMBIGUOUS:
            status = "mislabeled"
            findings.append(make_dataset_finding(
                "mislabeled_column", column, "medium",
                f"Column '{column}' is named like '{expected}' but its values look like '{predicted}' (confidence {confidence:.2f})",
            ))

        classification[str(column)] = {
            "predicted_class": predicted,
            "confidence": round(confidence, 4),
            "dominance": round(dominance, 4),
            "expected_from_name": expected,
            "status": status,
            "top_classes": {classes[i]: round(float(mean_probability[i]), 4) for i in np.argsort(mean_probability)[::-1][:3]},
        }

        if predicted in STRUCTURED_CLASSES and dominance >= 0.7 and confidence >= 0.6:
            class_index = {name: i for i, name in enumerate(classes)}
            off_type = np.array([
                label != predicted and not {label, predicted} <= AMBIGUOUS and probabilities[i, class_index[label]] >= FLAG_PROBABILITY
                for i, label in enumerate(per_value)
            ])
            mask = np.zeros(n, dtype=bool)
            mask[positions[off_type]] = True
            finding = make_finding(
                "semantic_inconsistency", column, "medium", mask,
                f"Column '{column}' mostly holds {predicted} values but some values look like a different type", n,
            )
            if finding:
                findings.append(finding)

    return classification, findings
