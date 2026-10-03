import json

import pandas as pd
import pytest

from module_1_profiling.profiler import profile_dataset
from module_2_cleaning.cleaner import clean_dataset, transform_features


def test_cleaning_writes_log_and_improves_quality_score(tmp_path):
    df = pd.DataFrame(
        {
            "customer_id": ["C-1", "C-2", "C-3", "C-4", "C-4"],
            "name": ["Alice Smith", "Alyce Smith", "Bob Jones", "Carl Doe", "Carl Doe"],
            "email": ["ALICE@EXAMPLE.COM", "alyce@example.com", "bob@example.com", "carl@example.com", "carl@example.com"],
            "age": ["30", "invalid", None, "45", "45"],
            "city": [" New York ", "new york", None, "Boston", "Boston"],
            "signup_date": ["2024/1/2", "2024-02-03", "bad-date", "2024-04-05", "2024-04-05"],
            "amount": [10, 20, 30, 40, 40],
        }
    )
    input_file = tmp_path / "messy.csv"
    df.to_csv(input_file, index=False)
    profile = profile_dataset(input_file, tmp_path / "profile")

    cleaned_path = clean_dataset(input_file, profile, tmp_path / "cleaned", fuzzy_threshold=0.85)
    cleaned = pd.read_csv(cleaned_path)
    log = json.loads((tmp_path / "cleaned" / "cleaning_log.json").read_text(encoding="utf-8"))

    assert len(cleaned) == 4
    assert cleaned.isna().sum().sum() == 0
    assert cleaned.loc[0, "email"] == "alice@example.com"
    assert cleaned.loc[0, "city"] == "new york"
    assert log["actions"]["exact_duplicates_removed"] == 1
    assert log["actions"]["fuzzy_duplicate_candidates"]
    assert log["schema_inference"]["amount"]["expected_range"] == {"min": 10.0, "max": 40.0}
    assert log["quality_score"]["after"]["overall"] > log["quality_score"]["before"]["overall"]


@pytest.mark.parametrize("strategy", ["knn", "regression", "iterative"])
def test_ml_imputation_strategies_fill_missing_numeric_values(tmp_path, strategy):
    df = pd.DataFrame({"feature_x": [1, 2, None, 4, 5], "feature_y": [2, 4, 6, 8, 10]})
    input_file = tmp_path / "numeric.csv"
    df.to_csv(input_file, index=False)
    profile = profile_dataset(input_file, tmp_path / f"profile-{strategy}")

    cleaned_path = clean_dataset(
        input_file,
        profile,
        tmp_path / f"cleaned-{strategy}",
        imputation_strategy=strategy,
    )

    assert pd.read_csv(cleaned_path).isna().sum().sum() == 0


def test_transform_features_scales_encodes_and_extracts_dates():
    df = pd.DataFrame(
        {
            "age": [20, 40],
            "city": ["Paris", "London"],
            "signup_date": ["2024-01-02", "2024-03-04"],
        }
    )

    transformed = transform_features(df)

    assert "signup_date" not in transformed.columns
    assert "signup_date_month" in transformed.columns
    assert "city_paris" in transformed.columns
    assert "city_london" in transformed.columns
    assert transformed["age"].mean() == pytest.approx(0.0)
