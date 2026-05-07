"""Build non-self retrieval evaluation fixtures from normalized records."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

from it_support.data import DownstreamUse
from it_support.data.candidate_sets import assert_allowed, is_allowed, unique_expected_domains


SOURCE_QUERY_PEER_GRADE = 1
SPECIFIC_TAG_PEER_GRADE = 2


def compact_text(*parts: str, limit: int = 2400) -> str:
    text = " ".join(part.strip() for part in parts if part and part.strip())
    text = " ".join(text.split())
    return text[:limit]


def answer_text(record: dict[str, Any]) -> str:
    accepted = record.get("accepted_answer") or {}
    top = record.get("top_answer") or {}
    accepted_text = accepted.get("answer_text") or ""
    top_text = top.get("answer_text") or ""
    if top_text and top_text != accepted_text:
        return compact_text(accepted_text, top_text, limit=1600)
    return compact_text(accepted_text, limit=1600)


def retrieval_document_text(record: dict[str, Any]) -> str:
    return compact_text(
        record.get("title", ""),
        record.get("question_text", ""),
        answer_text(record),
    )


def retrieval_query_text(record: dict[str, Any]) -> str:
    return compact_text(record.get("title", ""), record.get("question_text", ""), limit=900)


def normalized_tags(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    tags = []
    seen = set()
    for value in values or []:
        tag = str(value).strip().lower()
        if tag and tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return tags


def source_query_tags(record: dict[str, Any]) -> list[str]:
    return normalized_tags(record.get("query_tags"))


def question_tags(record: dict[str, Any]) -> list[str]:
    return normalized_tags(record.get("question_tags"))


def specific_question_tags(record: dict[str, Any]) -> list[str]:
    source_tags = set(source_query_tags(record))
    return [tag for tag in question_tags(record) if tag not in source_tags]


def project_retrieval_corpus_doc(record: dict[str, Any]) -> dict[str, Any]:
    assert_allowed(record, (DownstreamUse.RETRIEVAL,))
    return {
        "doc_id": record["record_id"],
        "record_id": record["record_id"],
        "source_family_id": record.get("source_family_id"),
        "source_run_id": record.get("source_run_id"),
        "source_url": record.get("source_url"),
        "site": record.get("site"),
        "question_id": record.get("question_id"),
        "title": record.get("title", ""),
        "document_text": retrieval_document_text(record),
        "primary_domain": record.get("primary_domain"),
        "domain_labels": unique_expected_domains(record),
        "query_tags": source_query_tags(record),
        "question_tags": question_tags(record),
        "license": record.get("license"),
        "commercial_posture": record.get("commercial_posture"),
        "commercial_reuse_allowed": False,
        "attribution_refs": record.get("attribution_refs", []),
        "downstream_allowed": {
            "retrieval": True,
            "evaluation": True,
            "answer_generation": False,
            "fine_tuning": False,
            "commercial_mode": False,
        },
    }


def project_retrieval_query_case(record: dict[str, Any], *, split: str) -> dict[str, Any]:
    assert_allowed(record, (DownstreamUse.EVALUATION,))
    return {
        "query_id": record["record_id"],
        "record_id": record["record_id"],
        "split": split,
        "source_family_id": record.get("source_family_id"),
        "source_run_id": record.get("source_run_id"),
        "source_url": record.get("source_url"),
        "site": record.get("site"),
        "question_id": record.get("question_id"),
        "title": record.get("title", ""),
        "query_text": retrieval_query_text(record),
        "expected_primary_domain": record.get("primary_domain"),
        "expected_domains": unique_expected_domains(record),
        "query_tags": source_query_tags(record),
        "question_tags": question_tags(record),
        "specific_question_tags": specific_question_tags(record),
        "license": record.get("license"),
        "commercial_posture": record.get("commercial_posture"),
        "commercial_reuse_allowed": False,
        "downstream_allowed": {
            "evaluation": True,
            "retrieval": False,
            "answer_generation": False,
            "fine_tuning": False,
            "commercial_mode": False,
        },
    }


def peer_group_key(record: dict[str, Any]) -> str:
    domain = str(record.get("primary_domain") or "unknown")
    tags = source_query_tags(record)
    return f"{domain}::{'|'.join(tags) if tags else 'untagged'}"


def build_relevance_judgments(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create graded non-self relevance judgments from deterministic metadata."""

    by_record_id = {str(row["record_id"]): row for row in records}
    source_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    specific_tag_groups: dict[tuple[str, str], list[str]] = defaultdict(list)

    for record_id, record in by_record_id.items():
        domain = str(record.get("primary_domain") or "unknown")
        for tag in source_query_tags(record):
            source_groups[(domain, tag)].append(record_id)
        for tag in specific_question_tags(record):
            specific_tag_groups[(domain, tag)].append(record_id)

    judgments_by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    def upsert(
        *,
        query_id: str,
        doc_id: str,
        grade: int,
        reason: str,
        shared_source_tags: set[str] | None = None,
        shared_specific_tags: set[str] | None = None,
    ) -> None:
        if query_id == doc_id:
            return
        existing = judgments_by_query[query_id].setdefault(
            doc_id,
            {
                "query_id": query_id,
                "relevant_doc_id": doc_id,
                "relevant_record_id": doc_id,
                "relevance_grade": grade,
                "match_reasons": [],
                "shared_source_query_tags": [],
                "shared_specific_question_tags": [],
            },
        )
        existing["relevance_grade"] = max(int(existing["relevance_grade"]), grade)
        if reason not in existing["match_reasons"]:
            existing["match_reasons"].append(reason)
        source_tags = set(existing["shared_source_query_tags"])
        source_tags.update(shared_source_tags or set())
        existing["shared_source_query_tags"] = sorted(source_tags)
        specific_tags = set(existing["shared_specific_question_tags"])
        specific_tags.update(shared_specific_tags or set())
        existing["shared_specific_question_tags"] = sorted(specific_tags)

    for record_id, record in by_record_id.items():
        domain = str(record.get("primary_domain") or "unknown")
        for tag in source_query_tags(record):
            for peer_id in source_groups[(domain, tag)]:
                upsert(
                    query_id=record_id,
                    doc_id=peer_id,
                    grade=SOURCE_QUERY_PEER_GRADE,
                    reason="same_primary_domain_and_source_query_tag",
                    shared_source_tags={tag},
                )
        for tag in specific_question_tags(record):
            for peer_id in specific_tag_groups[(domain, tag)]:
                upsert(
                    query_id=record_id,
                    doc_id=peer_id,
                    grade=SPECIFIC_TAG_PEER_GRADE,
                    reason="same_primary_domain_and_specific_question_tag",
                    shared_specific_tags={tag},
                )

    rows = []
    for query_id in sorted(judgments_by_query):
        query = by_record_id[query_id]
        for doc_id, judgment in sorted(
            judgments_by_query[query_id].items(),
            key=lambda item: (-int(item[1]["relevance_grade"]), item[0]),
        ):
            doc = by_record_id[doc_id]
            rows.append(
                {
                    **judgment,
                    "query_primary_domain": query.get("primary_domain"),
                    "doc_primary_domain": doc.get("primary_domain"),
                    "query_peer_group": peer_group_key(query),
                    "doc_peer_group": peer_group_key(doc),
                }
            )
    return rows


def positive_count_summary(qrels: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qrels:
        grouped[str(row["query_id"])].append(row)
    counts = [len(rows) for rows in grouped.values()]
    grade_two_counts = [
        sum(1 for row in rows if int(row["relevance_grade"]) >= SPECIFIC_TAG_PEER_GRADE)
        for rows in grouped.values()
    ]
    return {
        "queries_with_positives": len(counts),
        "min_positives_per_query": min(counts) if counts else 0,
        "median_positives_per_query": float(median(counts)) if counts else 0.0,
        "max_positives_per_query": max(counts) if counts else 0,
        "queries_with_grade_2_positives": sum(1 for count in grade_two_counts if count),
        "median_grade_2_positives_per_query": (
            float(median(grade_two_counts)) if grade_two_counts else 0.0
        ),
    }


def assert_retrieval_fixture_policy(
    *,
    records: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
) -> None:
    if any(not is_allowed(record, DownstreamUse.RETRIEVAL) for record in records):
        raise ValueError("All retrieval fixture corpus records must be retrieval-allowed")
    if any(is_allowed(record, DownstreamUse.ANSWER_GENERATION) for record in records):
        raise ValueError("Retrieval fixture refuses answer-generation-enabled records")
    if any("accepted_answer" in query or "top_answer" in query for query in queries):
        raise ValueError("Retrieval queries must stay question-only")
    if any(row["query_id"] == row["relevant_record_id"] for row in qrels):
        raise ValueError("Non-self retrieval qrels cannot point to the query record")
