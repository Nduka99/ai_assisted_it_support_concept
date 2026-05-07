from __future__ import annotations

from it_support.data import DownstreamUse, NormalizedArtifact, load_records
from it_support.retrieval.eval_fixtures import (
    SOURCE_QUERY_PEER_GRADE,
    SPECIFIC_TAG_PEER_GRADE,
    build_relevance_judgments,
    project_retrieval_query_case,
)


def _record(
    record_id: str,
    *,
    primary_domain: str = "network_connectivity",
    query_tags: list[str] | None = None,
    question_tags: list[str] | None = None,
) -> dict:
    return {
        "record_id": record_id,
        "title": f"title {record_id}",
        "question_text": f"question {record_id}",
        "primary_domain": primary_domain,
        "query_tags": query_tags or ["networking"],
        "question_tags": question_tags or ["networking"],
        "downstream_allowed": {
            "retrieval": True,
            "evaluation": True,
            "answer_generation": False,
            "fine_tuning": False,
            "commercial_mode": False,
        },
    }


def test_relevance_judgments_are_nonself_and_graded() -> None:
    rows = [
        _record("q1", question_tags=["networking", "bandwidth"]),
        _record("q2", question_tags=["networking", "bandwidth"]),
        _record("q3", question_tags=["networking", "vpn"]),
    ]

    qrels = build_relevance_judgments(rows)
    assert qrels
    assert all(row["query_id"] != row["relevant_record_id"] for row in qrels)

    q1_to_q2 = next(
        row for row in qrels if row["query_id"] == "q1" and row["relevant_record_id"] == "q2"
    )
    q1_to_q3 = next(
        row for row in qrels if row["query_id"] == "q1" and row["relevant_record_id"] == "q3"
    )
    assert q1_to_q2["relevance_grade"] == SPECIFIC_TAG_PEER_GRADE
    assert q1_to_q2["shared_specific_question_tags"] == ["bandwidth"]
    assert q1_to_q3["relevance_grade"] == SOURCE_QUERY_PEER_GRADE


def test_retrieval_query_projection_is_question_only() -> None:
    record = _record("q1")
    record["accepted_answer"] = {"answer_text": "do not leak this into query"}
    record["top_answer"] = {"answer_text": "or this"}

    query = project_retrieval_query_case(record, split="retrieval_dev_eval")

    assert query["query_id"] == "q1"
    assert "accepted_answer" not in query
    assert "top_answer" not in query
    assert "do not leak" not in query["query_text"]
    assert query["downstream_allowed"]["answer_generation"] is False


def test_current_retrieval_candidates_have_nonself_positives() -> None:
    result = load_records(
        NormalizedArtifact.STACK_EXCHANGE_RETRIEVAL_CANDIDATES,
        required_uses=(DownstreamUse.RETRIEVAL,),
    )

    qrels = build_relevance_judgments(result.records)
    query_ids = {row["query_id"] for row in qrels}
    grades = {row["relevance_grade"] for row in qrels}

    assert len(query_ids) == len(result.records)
    assert all(row["query_id"] != row["relevant_record_id"] for row in qrels)
    assert SOURCE_QUERY_PEER_GRADE in grades
    assert SPECIFIC_TAG_PEER_GRADE in grades
