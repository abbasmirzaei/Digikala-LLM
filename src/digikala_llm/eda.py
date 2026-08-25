"""Reproducible first-pass EDA for CSV/TSV product and review datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


ENCODINGS = ("utf-8-sig", "utf-8", "cp1256", "windows-1252", "latin1")


def detect_format(path: Path) -> tuple[str, str]:
    """Return a usable (encoding, delimiter) pair without mutating the source."""
    sample = path.read_bytes()[:100_000]
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            text = sample.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        try:
            dialect = csv.Sniffer().sniff(text, delimiters=",;\t|")
            return encoding, dialect.delimiter
        except csv.Error:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            return encoding, delimiter
    raise ValueError(f"Could not decode {path}") from last_error


def profile_dataset(path: Path) -> dict[str, Any]:
    encoding, delimiter = detect_format(path)
    df = pd.read_csv(path, encoding=encoding, sep=delimiter, low_memory=False)
    duplicate_rows = int(df.duplicated().sum())
    columns: list[dict[str, Any]] = []
    for name in df.columns:
        series = df[name]
        columns.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 2),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": [str(value)[:120] for value in series.dropna().head(3)],
            }
        )
    return {
        "source_file": str(path),
        "encoding": encoding,
        "delimiter": "\\t" if delimiter == "\t" else delimiter,
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": round(float(duplicate_rows / len(df) * 100), 2) if len(df) else 0.0,
        "columns": columns,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# First EDA report",
        "",
        f"- Source: `{report['source_file']}`",
        f"- Shape: **{report['rows']:,} rows × {report['columns_count']} columns**",
        f"- Encoding: `{report['encoding']}`",
        f"- Delimiter: `{report['delimiter']}`",
        f"- Duplicate rows: **{report['duplicate_rows']:,} ({report['duplicate_percent']}%)**",
        "",
        "## Columns",
        "",
        "| Column | dtype | Missing | Missing % | Unique |",
        "|---|---:|---:|---:|---:|",
    ]
    for column in report["columns"]:
        lines.append(
            f"| {column['name']} | {column['dtype']} | {column['missing_count']:,} | "
            f"{column['missing_percent']}% | {column['unique_count']:,} |"
        )
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_profile.json"
    markdown_path = output_dir / "dataset_profile.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a first-pass dataset profile.")
    parser.add_argument("dataset", type=Path, help="Path to a CSV or TSV file")
    parser.add_argument("--output", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error(f"Dataset not found: {args.dataset}")
    report = profile_dataset(args.dataset)
    json_path, markdown_path = write_report(report, args.output)
    print(f"Rows: {report['rows']:,}; columns: {report['columns_count']}")
    print(f"Reports: {json_path}, {markdown_path}")


if __name__ == "__main__":
    main()

