import json
from pathlib import Path

from digikala_llm.eda import detect_format, profile_dataset, profile_datasets, write_report


def test_detect_format_tolerates_utf8_character_split_at_sample_boundary(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "comments.csv"
    prefix = b"\xef\xbb\xbfid,body\n1," + (b"a" * (100_000 - 14))
    dataset.write_bytes(prefix + "ی\n".encode())

    assert dataset.read_bytes()[99_999:100_000] == "ی".encode()[:1]
    assert detect_format(dataset) == ("utf-8-sig", ",")


def test_chunked_product_profile_and_validation(tmp_path: Path) -> None:
    products = tmp_path / "digikala-products.csv"
    products.write_text(
        "id;title;Price;Rate;category\n"
        "1;one;100;90;A\n"
        "2;two;0;101;B\n"
        "2;three;-5;-1;A\n"
        ";four;bad;;B\n"
        "4;five;;;A\n",
        encoding="utf-8",
    )

    assert detect_format(products) == ("utf-8-sig", ";")
    report = profile_dataset(products, chunksize=2)

    assert report["rows"] == 5
    assert report["column_names"] == ["id", "title", "Price", "Rate", "category"]
    assert report["validation"]["id"] == {
        "missing_count": 1,
        "exact_full_row_duplicate_count": 0,
        "duplicate_id_excess_row_count": 1,
        "duplicated_unique_id_count": 1,
        "is_unique": False,
    }
    assert report["validation"]["Rate"]["outside_0_100_count"] == 2
    assert report["validation"]["Price"] == {
        "missing_count": 1,
        "zero_count": 1,
        "negative_count": 1,
        "non_numeric_count": 1,
    }
    assert report["validation"]["product_offer_variation"] == {
        "row_semantics": "products.id groups a product and may have multiple seller offers",
        "exact_duplicate_offer_count": 0,
        "product_ids_with_multiple_sellers": 0,
        "product_ids_with_different_prices": 1,
        "product_ids_with_conflicting_core_attributes": 1,
        "core_attributes": [
            "title_fa",
            "Category1",
            "Category2",
            "Brand",
            "Rate",
            "Rate_cnt",
            "sub_category",
        ],
    }
    assert "title" not in report["top_value_counts"]
    assert report["top_value_counts"]["category"] == {"A": 3, "B": 2}


def test_comments_and_join_validation(tmp_path: Path) -> None:
    products = tmp_path / "products.csv"
    comments = tmp_path / "comments.csv"
    products.write_text("id,Price,Rate\n1,10,50\n2,20,60\n2,30,70\n", encoding="utf-8")
    comments.write_text(
        "id,product_id,body,rate,recommendation_status,is_buyer\n"
        "10,1,good,5,recommended,True\n"
        "10,99,bad,6,not_recommended,False\n"
        "12,,ok,-1,,True\n"
        ",2,fine,3,recommended,\n",
        encoding="utf-8",
    )

    report = profile_datasets([products, comments], chunksize=2)
    comment_report = report["datasets"][1]

    assert report["chunksize"] == 2
    assert report["join_validation"]["relationship"] == "products.id <- comments.product_id"
    assert report["join_validation"]["orphan_comment_count"] == 1
    assert report["join_validation"]["orphan_product_ids_sample"] == ["99"]
    assert comment_report["validation"]["id"]["exact_full_row_duplicate_count"] == 0
    assert comment_report["validation"]["id"]["duplicate_id_excess_row_count"] == 1
    assert comment_report["validation"]["id"]["duplicated_unique_id_count"] == 1
    assert comment_report["validation"]["missing_product_id"] == 1
    assert comment_report["validation"]["rate"]["outside_0_5_count"] == 2
    assert comment_report["validation"]["recommendation_status"]["distribution"] == {
        "recommended": 2,
        "not_recommended": 1,
    }
    assert comment_report["validation"]["is_buyer"]["distribution"] == {
        "True": 2,
        "False": 1,
    }


def test_writes_json_and_markdown(tmp_path: Path) -> None:
    dataset = tmp_path / "products.csv"
    dataset.write_text("id,Price\n1,10\n", encoding="utf-8")
    report = profile_datasets([dataset], chunksize=1)

    json_path, markdown_path = write_report(report, tmp_path / "reports" / "eda")

    assert json.loads(json_path.read_text(encoding="utf-8"))["datasets"][0]["rows"] == 1
    assert "# Phase 1 EDA report" in markdown_path.read_text(encoding="utf-8")
