"""Restore the original source records from the prior augmented CSV.

The first 100 rows are the original project records; later rows were added by
this project's synthetic augmentation step.
"""

from pathlib import Path

import pandas as pd


SOURCE = Path("traffic_data.csv")
ORIGINAL = Path("traffic_data_original.csv")


def main() -> None:
    data = pd.read_csv(SOURCE)
    original = data.iloc[:100].copy()
    original.to_csv(ORIGINAL, index=False)
    original.to_csv(SOURCE, index=False)
    print(f"Restored {len(original)} original rows to {SOURCE}")
    print(f"Backup written to {ORIGINAL}")


if __name__ == "__main__":
    main()
