from __future__ import annotations

import sys
from pathlib import Path

from module_4_automation import load_config, run_pipeline


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CONFIG_FILE = ROOT / "config" / "pipeline.yaml"
INPUT_FILE = DATA_DIR / "sample_customer_data.csv"


def ensure_sample_data() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if INPUT_FILE.exists():
        return

    sample_data = """customer_id,name,email,age,city,signup_date,amount_spent,phone,notes
C-1001,Alice Johnson,alice@example.com,29,New York,2024-01-05,150.75,555-111-1111,Active customer
C-1002,Bob Smith,bob@example.com,,Chicago,2024-02-13,220.00,555-222-2222,Needs follow-up
C-1002,Bob Smith,bob@example.com,,Chicago,2024-02-13,220.00,555-222-2222,Needs follow-up
C-1003,Carol Davis,carol@domain.net,41,Los Angeles,2024-03-21,NaN,555-333-3333,VIP
C-1004,David Lee,david@example.com,invalid,Seattle,2024-05-10,340.50,555-444-4444,
C-1005,Eve Brown,eve@example.com,36,Miami,2024-06-18,120.00,555-555-5555,
C-1006,Frank White,frank@provider.com,57,,2024-07-12,90.25,555-666-6666,High value
C-1007,Grace Green,grace@example.com,28,Boston,not-a-date,88.10,555-777-7777,Verified
"""
    INPUT_FILE.write_text(sample_data, encoding="utf-8")


def main() -> int:
    ensure_sample_data()
    result = run_pipeline(load_config(CONFIG_FILE, base_dir=ROOT))

    for name, path in result.artifacts.items():
        print(f"{name}: {path}")
    print(f"Status: {result.status} (pipeline_ready={result.pipeline_ready})")
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
