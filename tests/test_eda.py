from pathlib import Path

from digikala_llm.eda import detect_format, profile_dataset


def test_detects_semicolon_and_profiles(tmp_path: Path) -> None:
    dataset = tmp_path / "products.csv"
    dataset.write_text("id;brand;price\n1;A;100\n2;;200\n2;;200\n", encoding="utf-8")

    assert detect_format(dataset) == ("utf-8-sig", ";")
    report = profile_dataset(dataset)

    assert report["rows"] == 3
    assert report["columns_count"] == 3
    assert report["duplicate_rows"] == 1
    assert report["columns"][1]["missing_count"] == 2

