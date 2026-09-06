"""Convert every raw dataset under datasets/ into plain CSV files under datasets-cleaned/.

Each dataset ships in its own format:
- amazongGames:   JSON-lines reviews (Video_Games_5.json)
- douban:         tab-separated, double-quoted text files (already have a header row)
- movieLense-1M:  "::"-separated .dat files with no header, latin-1 encoded
- movieLense-100k / movieLense-25M: already plain CSV, just copied through

Run from anywhere with:
    python "dataset loaders and cleaners/clean_datasets.py"
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "datasets-cleaned"

# Some douban review comments are long enough to trip the csv module's default limit.
csv.field_size_limit(10_000_000)

# Write with a BOM so Excel (the default way anyone previews these on Windows) picks
# up UTF-8 instead of guessing a local codepage and mangling non-ASCII text like
# douban's Chinese fields. utf-8-sig is a strict superset of utf-8 for every other
# reader (pandas, R, plain Python), so this is free for the all-ASCII datasets too.
OUTPUT_ENCODING = "utf-8-sig"

MOVIELENS_1M_SCHEMAS = {
    "movies.dat": ["MovieID", "Title", "Genres"],
    "ratings.dat": ["UserID", "MovieID", "Rating", "Timestamp"],
    "users.dat": ["UserID", "Gender", "Age", "Occupation", "Zip-code"],
}


def flatten(value: str) -> str:
    """Collapse embedded newlines to spaces so one record = one physical line.

    Free-text fields (review bodies, douban comments) contain literal newlines,
    which is valid CSV once quoted but makes the file look column-shifted in any
    tool that isn't a real CSV parser (Excel preview, Notepad, `head`, a quick
    eyeball scan).
    """
    return value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def convert_amazon_reviews(src: Path, dst: Path) -> None:
    """JSON-lines reviews -> CSV. Nested fields (e.g. style, image) are JSON-encoded."""
    fieldnames: list[str] = []
    seen: set[str] = set()
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for key in json.loads(line):
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as f_in, dst.open("w", encoding=OUTPUT_ENCODING, newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, restval="")
        writer.writeheader()
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            row = {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else flatten(value) if isinstance(value, str) else value
                )
                for key, value in record.items()
            }
            writer.writerow(row)
    print(f"  wrote {dst.relative_to(ROOT)}")


def convert_tab_separated(src: Path, dst: Path) -> None:
    """Quoted, tab-separated text (with its own header row) -> comma-separated CSV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8", newline="") as f_in, dst.open("w", encoding=OUTPUT_ENCODING, newline="") as f_out:
        reader = csv.reader(f_in, delimiter="\t", quotechar='"')
        writer = csv.writer(f_out)
        for row in reader:
            writer.writerow([flatten(field) for field in row])
    print(f"  wrote {dst.relative_to(ROOT)}")


def convert_movielens_dat(src: Path, dst: Path, header: list[str]) -> None:
    """"::"-separated, headerless, latin-1 .dat file -> comma-separated utf-8 CSV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="latin-1", newline="") as f_in, dst.open("w", encoding=OUTPUT_ENCODING, newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        for line in f_in:
            line = line.rstrip("\r\n")
            if not line:
                continue
            writer.writerow(line.split("::"))
    print(f"  wrote {dst.relative_to(ROOT)}")


def copy_as_is(src: Path, dst: Path) -> None:
    """Already plain CSV - copy through (streaming) so every dataset ends up under
    datasets-cleaned/, prefixing a BOM to match the rest of the output."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f_in, dst.open("wb") as f_out:
        f_out.write(b"\xef\xbb\xbf")
        shutil.copyfileobj(f_in, f_out)
    print(f"  wrote {dst.relative_to(ROOT)}")


def clean_amazon_games() -> None:
    src_dir, out_dir = DATASETS_DIR / "amazongGames", OUTPUT_DIR / "amazongGames"
    for src in src_dir.glob("*.json"):
        convert_amazon_reviews(src, out_dir / f"{src.stem}.csv")


def clean_douban() -> None:
    src_dir, out_dir = DATASETS_DIR / "douban", OUTPUT_DIR / "douban"
    for src in src_dir.glob("*.txt"):
        convert_tab_separated(src, out_dir / f"{src.stem}.csv")


def clean_movielens_1m() -> None:
    src_dir, out_dir = DATASETS_DIR / "movieLense-1M", OUTPUT_DIR / "movieLense-1M"
    for filename, header in MOVIELENS_1M_SCHEMAS.items():
        src = src_dir / filename
        if src.exists():
            convert_movielens_dat(src, out_dir / f"{src.stem}.csv", header)


def clean_already_csv(dataset_name: str) -> None:
    src_dir, out_dir = DATASETS_DIR / dataset_name, OUTPUT_DIR / dataset_name
    for src in src_dir.glob("*.csv"):
        copy_as_is(src, out_dir / src.name)


def main() -> None:
    steps = [
        ("amazongGames", clean_amazon_games),
        ("douban", clean_douban),
        ("movieLense-1M", clean_movielens_1m),
        ("movieLense-100k", lambda: clean_already_csv("movieLense-100k")),
        ("movieLense-25M", lambda: clean_already_csv("movieLense-25M")),
    ]
    for name, step in steps:
        print(f"Cleaning {name}...")
        step()
    print("Done.")


if __name__ == "__main__":
    main()
