import pandas as pd

from module_1_profiling.profiler import profile_dataset


def test_profile_dataset_contains_advanced_metadata(tmp_path):
    df = pd.DataFrame(
        {
            "customer_id": ["C-1001", "C-1002", "C-1003"],
            "name": ["Alice Johnson", "Bob Smith", "Carol Davis"],
            "email": ["alice@example.com", "bob@example.com", "carol@example.com"],
            "age": [29, "invalid", 41],
            "signup_date": ["2024-01-05", "bad-date", "2024-03-21"],
            "phone": ["555-111-1111", "555-222-2222", "555-333-3333"],
        }
    )

    file_path = tmp_path / "sample_data.csv"
    df.to_csv(file_path, index=False)

    report = profile_dataset(file_path)

    assert "metadata" in report
    assert report["metadata"]["semantic_types"]["email"] == "email"
    assert report["metadata"]["semantic_types"]["phone"] == "phone"
    assert "age" in report["metadata"]["mixed_type_columns"]
    assert "signup_date" in report["quality_checks"]["suspicious_columns"]
    assert "age" in report["quality_checks"]["suspicious_columns"]
