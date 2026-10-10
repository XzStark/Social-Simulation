from __future__ import annotations

from people.data_governance import impute_missing_rows


def test_impute_ffill_fills_missing() -> None:
    rows = [
        {"tick": 1, "policy_support": "0.5"},
        {"tick": 2, "policy_support": ""},
        {"tick": 3, "policy_support": "0.6"},
        {"tick": 4, "policy_support": "NA"},
    ]
    out = impute_missing_rows(rows, keys=["policy_support"], strategy="ffill")
    assert float(out[1]["policy_support"]) == 0.5
    assert float(out[3]["policy_support"]) == 0.6
