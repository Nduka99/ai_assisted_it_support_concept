"""Shared scoring helpers for retrieval evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def group_qrels(qrels: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels:
        query_id = str(row["query_id"])
        doc_id = str(row.get("relevant_doc_id") or row["relevant_record_id"])
        grouped[query_id][doc_id] = max(
            grouped[query_id].get(doc_id, 0),
            int(row["relevance_grade"]),
        )
    return dict(grouped)


def make_retrieved_items(
    *,
    query_id: str,
    docs: list[dict[str, Any]],
    scores: list[float],
    indices: list[int],
    qrels_for_query: dict[str, int],
    member_indices: list[int] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    retrieved = []
    seen = set()
    for score, index in zip(scores, indices, strict=False):
        if index < 0:
            continue
        source_index = member_indices[index] if member_indices is not None else index
        doc = docs[source_index]
        doc_id = str(doc["doc_id"])
        if doc_id == query_id or doc_id in seen:
            continue
        seen.add(doc_id)
        grade = int(qrels_for_query.get(doc_id, 0))
        retrieved.append(
            {
                "rank": len(retrieved) + 1,
                "doc_id": doc_id,
                "record_id": doc.get("record_id"),
                "score": round(float(score), 6),
                "relevance_grade": grade,
                "is_relevant": grade > 0,
                "primary_domain": doc.get("primary_domain"),
                "domain_labels": doc.get("domain_labels", []),
                "title": doc.get("title", ""),
                "source_url": doc.get("source_url"),
                "license": doc.get("license"),
                "commercial_posture": doc.get("commercial_posture"),
            }
        )
        if len(retrieved) >= top_k:
            break
    return retrieved


def dcg(grades: list[int], *, cutoff: int) -> float:
    return sum(
        ((2**grade - 1) / math.log2(rank + 1))
        for rank, grade in enumerate(grades[:cutoff], start=1)
    )


def query_metrics(
    retrieved: list[dict[str, Any]],
    qrels_for_query: dict[str, int],
    *,
    min_grade: int,
    cutoff: int,
) -> dict[str, Any] | None:
    relevant_grades = [
        grade for grade in qrels_for_query.values() if int(grade) >= min_grade
    ]
    if not relevant_grades:
        return None

    first_rank = None
    hit_at = {}
    for k in (1, 3, 5, 10):
        top = retrieved[: min(k, cutoff)]
        hit_at[f"hit_at_{k}"] = any(
            int(item.get("relevance_grade", 0)) >= min_grade for item in top
        )

    for item in retrieved[:cutoff]:
        if int(item.get("relevance_grade", 0)) >= min_grade:
            first_rank = int(item["rank"])
            break

    actual_grades = [
        int(item.get("relevance_grade", 0))
        if int(item.get("relevance_grade", 0)) >= min_grade
        else 0
        for item in retrieved[:cutoff]
    ]
    ideal_grades = sorted((int(grade) for grade in relevant_grades), reverse=True)
    ideal = dcg(ideal_grades, cutoff=cutoff)
    ndcg = dcg(actual_grades, cutoff=cutoff) / ideal if ideal else 0.0
    return {
        **hit_at,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "ndcg": ndcg,
        "first_relevant_rank": first_rank,
        "relevant_available": len(relevant_grades),
    }


def aggregate_metrics(
    prediction_rows: list[dict[str, Any]],
    qrels_by_query: dict[str, dict[str, int]],
    *,
    profile: str,
    relevance_scope: str,
    min_grade: int,
    cutoff: int,
) -> dict[str, Any]:
    per_query = []
    skipped = 0
    for row in prediction_rows:
        metrics = query_metrics(
            row["retrieved"],
            qrels_by_query.get(str(row["query_id"]), {}),
            min_grade=min_grade,
            cutoff=cutoff,
        )
        if metrics is None:
            skipped += 1
            continue
        per_query.append(metrics)

    total = len(per_query)
    if not total:
        return {
            "profile": profile,
            "relevance_scope": relevance_scope,
            "queries": 0,
            "queries_without_positives": skipped,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "hit_at_10": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
        }

    return {
        "profile": profile,
        "relevance_scope": relevance_scope,
        "queries": total,
        "queries_without_positives": skipped,
        "hit_at_1": round(sum(row["hit_at_1"] for row in per_query) / total, 4),
        "hit_at_3": round(sum(row["hit_at_3"] for row in per_query) / total, 4),
        "hit_at_5": round(sum(row["hit_at_5"] for row in per_query) / total, 4),
        "hit_at_10": round(sum(row["hit_at_10"] for row in per_query) / total, 4),
        "mrr": round(sum(float(row["mrr"]) for row in per_query) / total, 4),
        "ndcg": round(sum(float(row["ndcg"]) for row in per_query) / total, 4),
    }


def domain_metric_rows(
    prediction_rows: list[dict[str, Any]],
    qrels_by_query: dict[str, dict[str, int]],
    *,
    profile: str,
    relevance_scope: str,
    min_grade: int,
    cutoff: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[str(row.get("expected_primary_domain") or "unknown")].append(row)
    return [
        {
            "primary_domain": domain,
            **aggregate_metrics(
                rows,
                qrels_by_query,
                profile=profile,
                relevance_scope=relevance_scope,
                min_grade=min_grade,
                cutoff=cutoff,
            ),
        }
        for domain, rows in sorted(grouped.items())
    ]
