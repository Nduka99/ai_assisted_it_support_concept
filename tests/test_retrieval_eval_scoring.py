from __future__ import annotations

from it_support.retrieval.eval_scoring import (
    aggregate_metrics,
    group_qrels,
    make_retrieved_items,
)


def test_make_retrieved_items_excludes_query_source_doc() -> None:
    docs = [
        {"doc_id": "q1", "title": "self"},
        {"doc_id": "d1", "title": "peer"},
        {"doc_id": "d2", "title": "other"},
    ]
    qrels = {"d1": 2}

    retrieved = make_retrieved_items(
        query_id="q1",
        docs=docs,
        scores=[0.99, 0.8, 0.7],
        indices=[0, 1, 2],
        qrels_for_query=qrels,
        top_k=2,
    )

    assert [row["doc_id"] for row in retrieved] == ["d1", "d2"]
    assert retrieved[0]["rank"] == 1
    assert retrieved[0]["relevance_grade"] == 2


def test_aggregate_metrics_respects_min_grade_scope() -> None:
    qrels = group_qrels(
        [
            {"query_id": "q1", "relevant_doc_id": "d1", "relevance_grade": 1},
            {"query_id": "q1", "relevant_doc_id": "d2", "relevance_grade": 2},
        ]
    )
    predictions = [
        {
            "query_id": "q1",
            "expected_primary_domain": "network_connectivity",
            "retrieved": [
                {"rank": 1, "doc_id": "d1", "relevance_grade": 1},
                {"rank": 2, "doc_id": "d2", "relevance_grade": 2},
            ],
        }
    ]

    any_grade = aggregate_metrics(
        predictions,
        qrels,
        profile="p",
        relevance_scope="any_grade",
        min_grade=1,
        cutoff=10,
    )
    grade_2 = aggregate_metrics(
        predictions,
        qrels,
        profile="p",
        relevance_scope="grade_2",
        min_grade=2,
        cutoff=10,
    )

    assert any_grade["hit_at_1"] == 1.0
    assert any_grade["mrr"] == 1.0
    assert grade_2["hit_at_1"] == 0.0
    assert grade_2["hit_at_3"] == 1.0
    assert grade_2["mrr"] == 0.5
