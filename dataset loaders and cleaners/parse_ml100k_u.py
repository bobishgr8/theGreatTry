"""Convert the classic GroupLens ml-100k archive (u.data, tab-separated, no
header) into datasets/movieLense-100k/ratings.csv in the schema our loaders
expect (userId,movieId,rating,timestamp - same as ml-latest-small/ml-25m's
CSVs), so clean_datasets.py's existing movieLense-100k step (a plain copy)
picks it up unchanged.

The previous datasets/movieLense-100k/ratings.csv was actually ml-latest-small
(610 users, 9,724 items, 100,836 ratings) mislabeled as ml-100k - a different,
sparser dataset from the real benchmark (943 users, 1,682 items, exactly
100,000 ratings) that the UAIMC paper's Table 1 reports numbers on. The old
file is kept as ratings.csv.ml-latest-small.bak instead of being deleted.

Run from anywhere with:
    python "dataset loaders and cleaners/parse_ml100k_u.py"
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "movieLense-100k-u" / "u.data"
DST_DIR = ROOT / "datasets" / "movieLense-100k"
DST = DST_DIR / "ratings.csv"
BACKUP = DST_DIR / "ratings.csv.ml-latest-small.bak"


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"expected raw u.data at {SRC}")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    if DST.exists() and not BACKUP.exists():
        shutil.move(DST, BACKUP)
        print(f"backed up old (mislabeled) ratings.csv -> {BACKUP.relative_to(ROOT)}")

    n = 0
    with SRC.open(encoding="utf-8") as f_in, DST.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["userId", "movieId", "rating", "timestamp"])
        for line in f_in:
            line = line.rstrip("\r\n")
            if not line:
                continue
            user_id, item_id, rating, timestamp = line.split("\t")
            writer.writerow([user_id, item_id, rating, timestamp])
            n += 1

    print(f"wrote {n:,} ratings -> {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
