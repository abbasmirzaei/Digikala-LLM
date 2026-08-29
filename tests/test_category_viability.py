import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from digikala_llm.category_viability import build_inventory, write_inventory


def test_inventory_reconciles_hierarchical_scopes_and_prices(tmp_path: Path) -> None:
    products = tmp_path / "products.parquet"
    offers = tmp_path / "offers.parquet"
    comments = tmp_path / "comments.csv"
    pq.write_table(
        pa.table(
            {
                "product_id": [1, 2, 3],
                "category1": ["Care", "Care", "Other"],
                "category2": ["Sun", "Sun", None],
                "brand": ["A", "B", None],
            }
        ),
        products,
    )
    pq.write_table(pa.table({"product_id": [1, 1, 2, 3], "price_raw": [100, 90, 200, None]}), offers)
    comments.write_text(
        "id,title,body,created_at,rate,recommendation_status,is_buyer,product_id,advantages,disadvantages,likes,dislikes,seller_title,seller_code,true_to_size_rate\n"
        "1,t,good,1 فروردین 1400,5,recommended,True,1,,,,,,,\n"
        "2,t,nan,1 فروردین 1400,0,recommended,False,1,,,,,,,\n"
        "3,t,fine,1 فروردین 1400,6,recommended,True,2,,,,,,,\n",
        encoding="utf-8-sig",
    )
    rows, metadata = build_inventory(products, offers, comments)
    care_sun = next(
        row
        for row in rows
        if (row["scope_level"], row["category1"], row["category2"])
        == ("category1_category2", "Care", "Sun")
    )
    assert care_sun["product_count"] == 2
    assert care_sun["comment_count"] == 3
    assert care_sun["products_with_comments"] == 2
    assert care_sun["non_empty_body_count"] == 2
    assert care_sun["buyer_comment_count"] == 2
    assert care_sun["valid_positive_rating_count"] == 1
    assert care_sun["products_with_valid_historical_price"] == 2
    assert care_sun["historical_price_p50"] == 145
    assert all(metadata["reconciliation"].values())

    output = tmp_path / "inventory.csv"
    write_inventory(rows, output)
    output_repeat = tmp_path / "inventory-repeat.csv"
    write_inventory(rows, output_repeat)
    output_bytes = output.read_bytes()
    assert output_bytes == output_repeat.read_bytes()
    assert b"\r" not in output_bytes
    with output.open(encoding="utf-8", newline="") as source:
        assert len(list(csv.DictReader(source))) == 3
