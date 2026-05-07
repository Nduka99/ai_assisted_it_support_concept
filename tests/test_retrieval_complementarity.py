from __future__ import annotations

import pytest

from it_support.retrieval.complementarity import (
    compare_prediction_rows,
    diagnostic_upper_bound_rows,
    domain_complementarity_rows,
    first_rank_at_min_grade,
    outcome_count_rows,
)


def _prediction(
    profile: str,
    query_id: str,
    retrieved: list[dict],
    *,
    domain: str = "network_connectivity",
) -> dict:
    return {
        "profile": profile,
        "query_id": query_id,
        "split": "retrieval_dev_eval",
        "title": f"title {query_id}",
        "expected_primary_domain": domain,
        "expected_domains": [domain],
        "triage_predicted_domains": [domain],
        "retrieved": retrieved,
    }


def test_first_rank_at_min_grade_respects_grade_threshold() -> None:
    retrieved = [
        {"rank": 1, "doc_id": "d1", "relevance_grade": 1},
        {"rank": 2, "doc_id": "d2", "relevance_grade": 2},
    ]

    assert first_rank_at_min_grade(retrieved, min_grade=1) == 1
    assert first_rank_at_min_grade(retrieved, min_grade=2) == 2
    assert first_rank_at_min_grade(retrieved, min_grade=3) is None


def test_compare_prediction_rows_counts_grade_two_complementarity() -> None:
    baseline_rows = [
        _prediction(
            "bge",
            "q1",
            [
                {"rank": 1, "doc_id": "d0", "relevance_grade": 0, "title": "miss"},
                {"rank": 2, "doc_id": "d1", "relevance_grade": 2, "title": "hit"},
            ],
        ),
        _prediction(
            "bge",
            "q2",
            [{"rank": 1, "doc_id": "d2", "relevance_grade": 2, "title": "hit"}],
        ),
        _prediction(
            "bge",
            "q3",
            [{"rank": 1, "doc_id": "d9", "relevance_grade": 0, "title": "miss"}],
        ),
    ]
    challenger_rows = [
        _prediction(
            "qwen3",
            "q1",
            [{"rank": 1, "doc_id": "d1", "relevance_grade": 2, "title": "hit"}],
        ),
        _prediction(
            "qwen3",
            "q2",
            [{"rank": 1, "doc_id": "d0", "relevance_grade": 0, "title": "miss"}],
        ),
        _prediction(
            "qwen3",
            "q3",
            [{"rank": 1, "doc_id": "d8", "relevance_grade": 0, "title": "miss"}],
        ),
    ]
    qrels_by_query = {
        "q1": {"d1": 2},
        "q2": {"d2": 2},
        "q3": {"d3": 1},
    }

    rows = compare_prediction_rows(
        baseline_rows=baseline_rows,
        challenger_rows=challenger_rows,
        qrels_by_query=qrels_by_query,
    )
    eligible = [row for row in rows if row["is_grade_2_eligible"]]
    outcome_rows = outcome_count_rows(eligible, cutoffs=(1, 5))
    upper_rows = diagnostic_upper_bound_rows(eligible, cutoffs=(1, 5))
    domain_rows = domain_complementarity_rows(eligible, cutoffs=(1,))

    assert len(rows) == 3
    assert len(eligible) == 2
    assert rows[0]["outcome_at_1"] == "qwen3_only"
    assert rows[1]["outcome_at_1"] == "bge_only"
    assert rows[2]["is_grade_2_eligible"] is False
    assert {
        (row["cutoff"], row["outcome"]): row["queries"] for row in outcome_rows
    }[(1, "qwen3_only")] == 1
    assert upper_rows[0]["either_model_hit_rate"] == 1.0
    assert domain_rows[0]["bge_only_at_1"] == 1
    assert domain_rows[0]["qwen3_only_at_1"] == 1


def test_compare_prediction_rows_requires_same_query_set() -> None:
    baseline_rows = [_prediction("bge", "q1", [])]
    challenger_rows = [_prediction("qwen3", "q2", [])]

    with pytest.raises(ValueError, match="Prediction query sets differ"):
        compare_prediction_rows(
            baseline_rows=baseline_rows,
            challenger_rows=challenger_rows,
            qrels_by_query={},
        )
