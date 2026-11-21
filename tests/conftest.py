import pandas as pd
import pytest


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "customers.csv"
    pd.DataFrame(
        {
            "customer_id": [f"C-{1000 + i}" for i in range(12)],
            "email": [f"user{i}@example.com" for i in range(12)],
            "age": [20 + i for i in range(12)],
            "amount_spent": [100.0 + 7 * i for i in range(12)],
        }
    ).to_csv(path, index=False)
    return path
